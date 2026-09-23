# Reptile meta-training for few-shot robot post-training (cosmos_hs11)

`cosmos_hs11` = `cosmos_hs09` (data readers, episodic sampler, LoRA plumbing, held-out validation, all
unchanged) + a **Reptile** meta-trainer that replaces FOMAML/ANIL. Everything below the meta loop -- the
five embodiment datasets, `MetaEpisodeSpec`, `pack_samples_into_batch`, `meta_action_init.pt`, the
downstream LIBERO recipes -- is the hs09 code.

## 1. Why Reptile

The hs07/hs09 meta-initializations were optimized for **5 Adam steps** (FOMAML's first-order
approximation degrades with the inner horizon). Downstream is **1000-2000 steps of full fine-tuning**.
Measured consequences: the meta-learned theta sat on a "launch pad" whose zero-shot loss was *worse*
than the base, and 2000-step full FT neither undid nor used the meta delta (cos(FT move, delta) = 0).

Reptile (Nichol, Achiam & Schulman 2018) optimizes a different thing. One meta-iteration is

```
theta_tilde = k standard optimizer steps on one embodiment's K demonstrations, starting from theta
theta      <- theta + eps * (theta_tilde - theta)
```

whose stationary point is the parameter-space **centroid of the k-step solutions** of all embodiments
("an init from which ordinary fine-tuning is short everywhere"). k can be as long as the compute allows,
there is no query set and no meta-gradient, and -- decisive here -- the meta step is a plain interpolation,
so the model's own **FSDP-sharded fp32 parameters are the fast weights** and the ordinary training stack
runs the inner loop. That is what makes **full-parameter** meta-training of `moe_gen` (1.4B) affordable
(FOMAML needed per-rank distinct fast weights plus a query gradient at the adapted point).

## 2. Two modes

| | `theta_mode = "full"` (default) | `theta_mode = "lora"` |
| --- | --- | --- |
| theta = trainable set | moe_gen + time_embedder + vae2llm + llm2vae + k_norm_und_for_gen + action heads (1.42B) | LoRA(r32, alpha 64, q/k/v/o + mlp up/down) + heads + time_embedder (38M) |
| = downstream recipe | `action_policy_libero_edge` (full FT) | `action_policy_libero_lora_edge` (LoRA) |
| inner optimizer | FusedAdam lr **5e-5**, wd 0.05, heads x1 | FusedAdam lr **1e-4**, wd 0, heads x1 |
| TOML | `action_reptile_meta_edge.toml` | `action_reptile_meta_lora_edge.toml` |
| saved theta | DCP `checkpoints/iter_XXXXXXXXX` (~24 GB: 13.5 GB weights + Adam moments) + `meta_action_init*.pt` (heads) | DCP + `meta_action_init*.pt` (heads + LoRA + TE) |
| downstream | `action_policy_libero_10_edge_reptileinit.toml` (`REPTILE_CKPT_PATH` + `META_ACTION_INIT_PATH`) | hs09 TOMLs unchanged: `action_policy_libero_10_lora_edge_metainit.toml` or `action_policy_libero_10_edge_metainit.toml` |

The two modes are the same script and experiment; the TOML changes `[model] lora_*`,
`[optimizer] keys_to_select / lr / weight_decay`, `[checkpoint] keys_to_skip_loading` (+`"lora_"`) and
`[custom.meta] theta_mode`. **Rule: theta == the downstream trainable set**, so the inner loop is a miniature of the
post-training it prepares for (same optimizer, same LR, same loss).

## 3. Files

| piece | path |
| --- | --- |
| Reptile trainer | `cosmos_framework/scripts/train_action_reptile.py` |
| theta bookkeeping (shard copy, interpolation, group norms, export) | `cosmos_framework/data/generator/action/meta/reptile_meta.py` |
| rank-synchronized episodes (same embodiment/demos on every rank, sharded windows) | `SynchronizedMetaEpisodeIterableDataset` / `build_reptile_episode_loader` in `meta/episodic_sampler.py` |
| experiment | `configs/base/experiment/action/meta/action_reptile_meta_edge.py` |
| run TOMLs | `examples/toml/sft_config/action_reptile_meta_edge{,_lora,_joint,_smoke}.toml` |
| launcher | `examples/launch_reptile_meta_edge.sh` (`TOML_FILE=...` selects the mode; MASTER_PORT 50016) |
| downstream (full mode) | `examples/toml/sft_config/action_policy_libero_10_edge_reptileinit.toml`, `examples/launch_sft_action_policy_libero_10_edge_reptileinit.sh` |
| tests | `meta/reptile_meta_test.py`, `test_synchronized_dataset_*` in `meta/episodic_sampler_test.py` |

Mechanics worth knowing:

* **All ranks train the same embodiment** (FSDP requires identical weights); the K demonstrations are shared
  and their windows sharded `[rank::world]`. Every item carries `spec_id`; the trainer all-gathers it and
  aborts on desync instead of deadlocking. A rank that fails to decode yields `failed=True` and the whole
  episode is skipped on all ranks.
* The inner optimizer is built by the framework (`model.init_optimizer_scheduler`), so `keys_to_select`,
  `lr_multipliers`, weight decay and FusedAdam's fp32 master weights behave exactly as in post-training.
  Its Adam state (and FusedAdam's per-group `step`) is reset every episode (`inner_reset_optimizer`), and the
  LR ramps linearly over `inner_warmup_steps` then stays constant.
* Checkpoints go through `DistributedCheckpointer.save`, i.e. the standard DCP layout with
  `latest_checkpoint.txt`: **re-running the same command auto-resumes** (model = theta, iteration), and the
  DCP is a valid `load_path` for any post-training recipe. The inner Adam moments are cleared before saving.
* `spec_offset = start_iter` on resume so the episode stream does not repeat.

## 4. Hyper-parameters and why (4 x A6000/A40, 1000 meta-iterations)

| knob | full | lora | why |
| --- | --- | --- | --- |
| `k_shot` / `windows_per_demo` | 8 / 16 -> 128 support windows | same | 32 windows per rank per step = one pass over the support set per inner step; ~20 passes over K demos mirrors the 3-demo LIBERO post-training (~100+ epochs) |
| `q_query` | 4 x 16 = 64 | same | diagnostics only (zero-shot vs adapted query loss) |
| `max_samples_per_batch` (per-rank inner batch) | 32 | 32 | 128 windows/step globally; VRAM ~25 GB/rank in full mode |
| `inner_steps` k | 20 | 20 | inner ~6 s/step full -> ~2 min inner per meta-iteration |
| inner lr | 5e-5 | 1e-4 | = the downstream recipe of the mode |
| `inner_warmup_steps` | 3 | 3 | Adam's first steps are sign-like; a short ramp avoids a jump at every episode start |
| `inner_grad_clip` | 1.0 | 1.0 | as downstream |
| `meta_lr` eps | 0.5 -> 0.05 linear | same | Reptile anneals eps toward 0; 0.5 keeps the serial single-embodiment updates from oscillating |
| `meta_optimizer` | sgd | sgd | the original algorithm; `"adam"` (Adam on the displacement) is available |
| `loss_mode` | total | total | the recipe's objective, 10 x vision + 10 x action (`loss_scale` and `action_loss_weight` of `EDGE_MODEL_CONFIG`); `"action"` for an ANIL-style variant |
| `max_iter` | 1000 | 1000 | ~3 min/iter full (~1 min encode + query evals, ~2 min inner) -> **~45-50 h**; ~2 min/iter lora -> ~35 h |
| `save_iter` | 250 | 250 | DCP snapshots are ~24 GB in full mode (`get_optimizer_state_dict` materializes the Adam moments even though they are reset) |

Cost lever: `inner_steps` (linear), `windows_per_demo` (encode + inner), `eval_query_every` (2 forwards),
`max_iter`. eps anneals **at `max_iter`** -- change `max_iter`, do not stop a run early.

## 5. Controls and what to compare

* **Joint multi-task control** (`action_reptile_meta_edge_joint.toml`): `inner_steps=1, meta_lr=1.0,
  meta_lr_min_ratio=1.0, inner_reset_optimizer=false` -- with k=1 and eps=1 Reptile IS sequential multi-task
  training on the same data stream. Its checkpoint is "robot pretraining without meta-learning"; the meta
  contribution is Reptile minus this.
* **Fresh baseline**: the hs08 / hs09 full-FT baseline from the mid-trained base.
* Compare on `val/flow_matching_loss_action` (iteration 0 = the init itself) and closed-loop SR at matching
  iterations with >= 20 trials/task.

Logs / wandb: `query_loss_zero_shot` vs `query_loss` (adapted) and `query_loss_gain`; `adapt_delta/<group>`
(how far k steps move each group), `meta_step_norm` (eps x displacement), `theta_norm/<group>`,
`support_loss_first -> last`, `inner_grad_norm`, timings `t_data/t_encode/t_inner/t_eval/t_meta`.
Unlike FOMAML, a **falling zero-shot loss is expected** (theta moves toward the solutions, not to a launch pad).

## 6. Commands

```bash
cd ~/project/cosmos_hs11/packages/cosmos3
export ROBOT_FEWSHOT_ROOT=$COSMOS_STORAGE/data/robot_fewshot
export BASE_CHECKPOINT_PATH=$COSMOS_STORAGE/checkpoints/Cosmos3-Edge
export LIBERO_ROOT=$COSMOS_STORAGE/data/LIBERO_LeRobot_v3/libero_10

# (0) 2-GPU smoke (3 meta-iterations)
TOML_FILE=examples/toml/sft_config/action_reptile_meta_edge_smoke.toml NPROC_PER_NODE=2 sr 2 48 examples/launch_reptile_meta_edge.sh

# (1) Reptile meta-training, FULL mode (default TOML), 4 GPUs
NPROC_PER_NODE=4 sr 4 48 examples/launch_reptile_meta_edge.sh
#     LoRA mode instead:
TOML_FILE=examples/toml/sft_config/action_reptile_meta_lora_edge.toml NPROC_PER_NODE=4 sr 4 48 examples/launch_reptile_meta_edge.sh
#     joint multi-task control:
TOML_FILE=examples/toml/sft_config/action_reptile_meta_edge_joint.toml NPROC_PER_NODE=4 sr 4 48 examples/launch_reptile_meta_edge.sh

# (2) downstream, FULL mode: full-FT post-training warm-started from theta
R=$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_meta/reptile_meta/edge_reptile_full_k8w16_s20_eps0.5_seed42
export REPTILE_CKPT_PATH=$R/checkpoints/iter_000001000 META_ACTION_INIT_PATH=$R/meta_action_init_iter_001000.pt
NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_libero_10_edge_reptileinit.sh

# (2') downstream, LoRA mode: the hs09 recipes, pointed at the Reptile .pt
export META_ACTION_INIT_PATH=$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_meta/reptile_meta/edge_reptile_lora_r32_k8w16_s20_eps0.5_seed42/meta_action_init_iter_001000.pt
NPROC_PER_NODE=2 sr 2 48 examples/launch_sft_action_policy_libero_10_lora_edge_metainit.sh      # LoRA post-training
NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_libero_10_edge_metainit.sh           # full FT with the meta LoRA frozen

# (3) closed-loop eval as before (examples/eval_libero_closed_loop.sh)
```

## 7. How to change things

| you want | change |
| --- | --- |
| a longer / shorter inner fine-tune | `[custom.meta] inner_steps` (cost is linear); keep `inner_warmup_steps` <= steps/4 |
| more demos per episode (closer to 30-demo LIBERO) | `k_shot` (and `windows_per_demo`); `k_shot * windows_per_demo / 4 ranks` must be a multiple of `max_samples_per_batch` for exactly one pass per step |
| run on 2 GPUs | `action_reptile_meta_edge_2gpu.toml` (shard 2, 64 windows/rank = the same 128-window inner batch, identical optimisation; ~1.7x wall-clock). Downstream: `action_policy_libero_10_edge_reptileinit_2gpu.toml` (128/rank x 2 = 256; OOM fallback in its comments: bs 64 + `trainer.grad_accum_iter=2`) |
| less VRAM | `max_samples_per_batch` (per rank) down -- with 2 steps per support pass the optimisation changes slightly (2x more, noisier Adam steps); prefer fewer `windows_per_demo` if one pass per step matters |
| a different downstream recipe | make `[optimizer] keys_to_select / lr / weight_decay / lr_multipliers` equal to that recipe -- theta must equal its trainable set |
| stronger / weaker meta step | `meta_lr` (eps in (0, 1]); `meta_batch_embodiments > 1` averages several embodiments per step (x cost) |
| action-only inner objective | `loss_mode = "action"` (then `llm2vae` gets no gradient; consider removing it from `keys_to_select`) |
| export more groups into the .pt | `theta_mode = "lora"` exports LoRA + time_embedder; in full mode they live in the DCP |
| run shorter | change `[trainer] max_iter` (eps schedule is tied to it) |

## 7b. VRAM and wall-clock (full mode)

Per-GPU memory is dominated by the FSDP shards of persistent state, all fp32 (`fsdp_master_dtype=float32`):

| state | total | per GPU, shard 4 | per GPU, shard 2 |
| --- | --- | --- | --- |
| master weights, 3.37B params (both towers) | 13.5 GB | 3.4 GB | 6.7 GB |
| grads of the 1.42B trainable set | 5.7 GB | 1.4 GB | 2.8 GB |
| Adam m, v | 11.4 GB | 2.8 GB | 5.7 GB |
| Reptile theta copy | 5.7 GB | 1.4 GB | 2.8 GB |
| **persistent total** | **36.3 GB** | **9.1 GB** | **18.2 GB** |

On top: activations with full activation checkpointing (roughly 0.1-0.15 GiB per 256-res window, so 3-5 GiB at 32 windows/rank
and 6-10 GiB at 64/rank), the VAE encode of the episode, and 2-3 GiB CUDA/NCCL overhead. **Measured** (2x A6000, shard 2 x 64
windows, 2026-09-21): 25.5 GiB peak allocated / 31.3 GiB reserved per GPU, 150 s per meta-iteration in steady state
(inner loop 121 s = 20 x 6.0 s, VAE encode 28 s, eval 2 s; the first iteration adds ~90 s of DataLoader warm-up), i.e.
1000 iterations ~ 42 h. Shard 4 x 32 windows is estimated at 17-22 GiB/GPU and ~1.5-2 min/iteration. The per-iteration log
line prints `peak mem <alloc> GiB alloc / <reserved> GiB reserved` (also `mem_peak_alloc_gb` / `mem_peak_reserved_gb` in the
jsonl and wandb), so the real footprint is visible after the first iteration.

The downstream *full-FT* post-training with 128 windows/rank on 2 GPUs has no theta copy but keeps an EMA copy and a 2-4x
larger activation set; it OOMed on 44 GiB nodes in cosmos_hs09. Use `action_policy_libero_10_edge_reptileinit_2gpu.toml`
with the bs 64 + `grad_accum_iter=2` override (same global batch 256, same optimisation) when that happens.

Wall-clock scales with `inner_steps x windows per step`; the inner loop is ~80% of an iteration, so `inner_steps` is the
lever. Read `iter_time` from the first ~10 iterations (skip iteration 1) and multiply by `max_iter`.

## 8. Caveats

* Serial Reptile sees one embodiment per meta step; with eps 0.5 theta zig-zags between embodiments and the
  annealing of eps is what settles it. `meta_batch_embodiments=5` removes the zig-zag at 5x the cost.
* Group norms (`adapt_delta/*`, `theta_norm/*`) assume pure FSDP sharding (`replicate_degree=1`); under
  HSDP they are over-counted (diagnostics only, the meta step itself is exact).
* A full-mode DCP is ~24 GB; 1000 iterations with `save_iter=250` write ~100 GB. Downstream reads only its `model/` part.
* `bob` (broken RoCE) is excluded automatically for single-node runs via `NCCL_IB_DISABLE=1`; the
  hs09 `_sft_launcher_common.sh` shard-vs-NPROC guard applies to the downstream launchers.
