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

* **Ordinary source-SFT control of the edge2 run** (`action_reptile_meta_edge2_joint_sft.toml`, launcher
  `launch_source_sft_edge2_joint.sh`, port 50054, 2 GPUs): the same ablation idea applied to the edge2 recipe -- identical
  sampler (16 demos x 16 windows = 256 per iteration, embodiment uniform per iteration, shared head row 31), theta,
  optimizer, shard 2 x 128, 1000 iterations; only `inner_steps=1, inner_warmup_steps=0, inner_reset_optimizer=false,
  meta_lr=1.0, meta_lr_min_ratio=1.0`, so each iteration is one FusedAdam step and `theta <- theta_tilde` (Reptile k=1 ==
  joint training). Data-matched to edge2 (same 1000 episodes) but 10x fewer Adam steps; `trainer.max_iter=10000` gives the
  step-matched variant. Downstream: `action_policy_libero_10_edge_sftinit.toml` (= the reptileinit recipe with a distinct
  name; REPTILE_CKPT_PATH / META_ACTION_INIT_PATH point at the control run). Compares "trunk+heads pretrained on the 5
  sources by plain SFT" against "by Reptile" under the identical LIBERO recipe, isolating the meta-learning rule.
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

**Full-data post-training on the Reptile init** (`action_policy_libero_10_edge_reptileinit_data_all.toml`, launcher
`..._reptileinit_data_all.sh`, port 50049): identical to the 3-demo recipe except (1) `[dataloader_train].episode_subset_path
= "libero_10_all_episodes.json"` = all 379 episodes listed as a fixed subset (omitting the key would let the dataset carve
out its own seeded 1 % val split and train on 375), and (2) validation off (`run_validation=false`, no `[dataloader_val]`):
nothing is held out, so judge by rollout success on checkpoints picked by iteration. Same 2000-iteration schedule
(~5 epochs instead of ~67); lengthen `trainer.max_iter` together with `scheduler.cycle_lengths` if wanted, in the
fresh-init baseline too.

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

**v43 (2026-10-06): demonstration-uniform sampling.** `action_policy_libero_10_edge_reptileinit_v43.toml` /
`launch_sft_action_policy_libero_10_edge_reptileinit_v43.sh` (port 50063) = the user's v3 (head lr x2) plus
`[dataloader_train].episode_balanced_sampling = true`. The default streaming loader (`ActionIterableShuffleDataset`)
visits every window of every demonstration once per epoch, so a demonstration's share of the samples is proportional to
its window count; `ActionEpisodeUniformIterableDataset` instead draws one of the 30 demonstrations uniformly and then a
window uniformly inside it (with replacement, independent seeded stream per rank x worker), which makes the expected
per-demonstration and per-task sample counts equal. Knob = `get_action_libero_sft_dataset(episode_balanced_sampling=...)`
(needs `iterable_shuffle=True`), routed from the TOML to the nested libero dataset node by `toml_config_helper`; the
experiment default is False, so no other recipe changes. Validation keeps the window-uniform held-out pass. Tests:
`datasets/action_sft_dataset_test.py`, `toml_config/libero_sampling_toml_test.py`.

## 8. Post-training variants on the Reptile init: DINOv2 REPA and masked V-JEPA 2.1

Both add an auxiliary representation objective to the plain `action_policy_libero_10_edge_reptileinit.toml` recipe; the
TOMLs differ from it only in `[job].experiment/name`, a `[model.repa]` block and `"repa_"` in `keys_to_skip_loading`, so
the three arms (plain / REPA / masked JEPA) isolate the effect of the extra loss. Code ported from cosmos_hs10 (REPA
module, `docs/action_policy_libero_repa_vjepa.md`) and cosmos_hs12 (`docs/action_policy_libero_masked_jepa.md`) via the
cosmos_hs09_2 merge; experiment `action_policy_libero_edge_repa` (+ `repa_` in keys_to_select, `keep_native_video`).

| variant | TOML / launcher | loss added to 10*fm_vision + 10*fm_action |
| --- | --- | --- |
| DINOv2 REPA (hs10 v7) | `action_policy_libero_10_edge_reptileinit_repa_dinov2.toml`, `launch_sft_action_policy_libero_10_edge_reptileinit_repa_dinov2.sh` (port 50024) | `5.0 * (1 - cos(MLP(h_8), avgpool(DINOv2-ViT-B/14(frames 1..16))))` on the predicted video tokens |
| DINOv2 REPA, ramped weight (v3) | `action_policy_libero_10_edge_reptileinit_repa_dinov2_v3.toml`, `launch_sft_action_policy_libero_10_edge_reptileinit_repa_dinov2_v3.sh` (port 50028) | the same term with `5.0 * min(1, step / 200)` instead of `5.0` (`[model.repa] loss_weight_warmup_steps = 200`, logged as `repa_weight`): 0 at iteration 0, so the fresh REPA head does not dominate the first updates of the Reptile init (measured: with the constant weight it was ~50% of the loss in the first ~100 iterations vs ~28% in fresh-init hs10 v7; identical from iteration 200 on) |
| masked JEPA (hs12 dense) | `action_policy_libero_10_edge_reptileinit_masked_jepa.toml`, `launch_sft_action_policy_libero_10_edge_reptileinit_masked_jepa.sh` (port 50025) | `0.5 * ramp(step/200) * (mean_masked|p-y| + 0.25 mean_visible|p-y|)`, p = MLP(h_8) of a pixel-masked re-encoded clip, y = frozen V-JEPA 2.1 ViT-B tokens |

```bash
JOB=$COSMOS_STORAGE/outputs/cosmos3_action_meta/reptile_meta/<reptile run name>
export LIBERO_ROOT=<libero_10 LeRobot dir>
export REPTILE_CKPT_PATH=$JOB/checkpoints/iter_000001000
export META_ACTION_INIT_PATH=$JOB/meta_action_init_iter_001000.pt
NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_libero_10_edge_reptileinit_repa_dinov2.sh    # HF_HOME: facebook/dinov2-base
NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_libero_10_edge_reptileinit_masked_jepa.sh    # COSMOS_STORAGE: vjepa2_1 ckpt
```

Notes: the REPA teachers are frozen and never checkpointed; `repa_head` (3-layer MLP) is the only new trainable module
and starts fresh. Both recipes ship the native uint8 frames to the teacher (`keep_native_video`), which costs a few GB
of host/GPU memory; 4 x 64 windows fits the 44 GiB cards, the 2-GPU x 128 layout only the 48 GiB A6000s. Watch
`val/repa_cos_sim_centered` (raw `repa_cos_sim` saturates near 0.98) and, for masked JEPA, `jepa_masked_loss` vs
`jepa_visible_loss` and `jepa_centered_cos`. The meta trainer's inner loop does not support either objective (it calls
`training_step_from_inputs` without the raw batch), so these are post-training-only losses.

**v5 (2026-09-29)** = v2 + less pooling: `[model.repa].target_subgrid_thw = [1, 2, 2]` (each MoT token predicts a 2x2
block of the 4x10x10-per-view DINOv2 grid; see docs/action_policy_libero_repa_vjepa.md section 9). Launcher
`examples/launch_sft_action_policy_libero_10_edge_reptileinit_repa_dinov2_v5.sh` (port 50048).

## 8b. REPA inside the Reptile inner loop (v11)

`examples/toml/sft_config/action_reptile_meta_edge_v11.toml` + `examples/launch_reptile_meta_edge_v11.sh` (port 50082)
move the DINOv2 distillation from the post-training loss into the meta-training objective. With `L_base` the recipe's
`10 x fm_vision + 10 x fm_action`, `T` the frozen DINOv2 ViT-B/14 and `P_phi = net.repa_head` the REPA projector:

```
L_inner = L_base + lambda_meta * (1 - cos(P_phi(h_theta^(l)), sg[T(x)]))      psi = (theta, phi)
psi' = U^k_{L_inner}(psi)            k inner FusedAdam steps on one embodiment's K demos (as before)
psi  <- psi + eps * (psi' - psi)     the unchanged Reptile step, now on the backbone AND the projector
```

* **What is tuned in the TOML**: `[model.repa] layer_index` (MoT block `l`, 1..28) and `loss_weight` (`lambda_meta`;
  `loss_weight_warmup_steps` ramps it 0 -> `loss_weight` over the first N META-iterations). Teacher, input size,
  objective, projector and adapter are the hs10 v7 settings; `num_views = 1`: the canvas is one token grid for the
  alignment head (`avgpool` adapts to the real grid of every batch, so the 8x8 / 8x10 / 6x10 token grids of the
  "256" buckets all work).
* **Camera views** (`[model.repa.view_layouts]`, `model/generator/repa/view_layouts.py`): RT-1, Bridge (image_0),
  RoboMIND-UR / -Franka (`*_1rgb`) ship one camera, so their canvas is one 224 px teacher image. MolmoAct2-YAM ships
  the `compose_multiview` 2x2 composite (top camera over the two half-sized wrist cameras, 540x640), which the stock
  width split cannot separate; with `molmoact2_yam = "primary_over_two"` the teacher crops the native canvas into the
  three views, encodes each at 224 px (a full 16x16 DINOv2 grid per view) and re-assembles the grids in the composite
  layout (24x16 cells: top 16x16, wrists 8x8 each) before the usual pooling onto the window's MoT grid. Needs
  `num_views = 1`, an avgpool-type adapter, no student upsampler; such embodiments keep their camera resolution in
  the loader (`native_video_full_res`). The same layout fits RoboMIND `concat_view` (top | left, right) should a 3-camera
  export be used. Samples without a layout entry use the stock path.
* **What the trainer does** (`train_action_reptile.py`, only when `model.config.repa.enabled`): checks
  `loss_mode = "total"` and that `"repa_"` is in `optimizer.keys_to_select` (phi must be part of theta, group
  `repa_head`); switches the episode loader to `keep_native_video` and shrinks the native clips to
  `teacher_input_size` in the workers (`native_video_size`; both overridable from `[custom.meta]`); runs the frozen
  teacher ONCE per meta-episode (like the VAE encode) and hands the tokens to every inner step and query evaluation
  through `training_step_from_inputs(..., repa_teacher_tokens=...)`. `objective = "masked_prediction"` is rejected
  (it needs the raw batch every step).
* **Logging**: `support_repa_loss_first/last`, `support_repa_cos_centered_last`, `repa_weight`, `support_loss_fm_last`,
  `query_repa_loss(_zero_shot)`, `query_repa_cos_centered`, `query_loss_fm(_zero_shot)`, `adapt_delta/repa_head`,
  `theta_norm/repa_head`. `support_loss_*` / `query_loss*` are the TOTAL incl. the weighted REPA term -> compare the
  `*_fm` keys with a v4 run.
* **Scale**: the inner support loss of v4 sits at ~1-3 (x10-scaled fm terms); a fresh projector starts at
  `1 - cos ~ 1`, so `loss_weight = 5` (the post-training value) would dominate the first inner steps. v11 starts at
  1.0 with a 50-meta-iteration ramp; the meta-learned projector makes the term shrink over the run.
* **Downstream**: the DCP contains `net.repa_head.*`. All existing downstream TOMLs skip `"repa_"` at load (fresh
  projector; the plain `reptileinit` recipe has no head at all), so they run unchanged on a v11 checkpoint. To
  continue with the meta-learned projector, remove `"repa_"` from the post-training TOML's `keys_to_skip_loading` and
  keep its `[model.repa]` `layer_index` / `teacher` / `projector_*` identical to v11 (`num_views` / grids follow the
  LIBERO canvas there).
* **Controls**: `action_reptile_meta_edge2_joint_sft.toml` + the same `[model.repa]` block = "source-SFT + REPA"
  (k = 1, eps = 1), which separates the meta-learning contribution from plain REPA pretraining on the source data.
* Memory: v4 peaked at 28.8 GiB alloc / 33.8 GiB reserved per GPU (4 x 64 windows); v11 adds the teacher (~0.2 GB),
  the cached tokens (~0.4 GB bf16 per 64 windows), the shrunk clips (~0.16 GB) and the head. Not GPU-smoked yet.

## 9. Cosmos3-Nano

The same trainer and knobs run on the Nano tier (Qwen3-VL-8B MoT: hidden 4096, 36 blocks, 15.75B params, generation
tower 6.95B) through the experiment `action_reptile_meta_nano` (`configs/.../meta/action_reptile_meta_nano.py`), which
mirrors `action_reptile_meta_edge` plus the few-shot Nano recipe's model deltas (`loss_scale` 10, fresh diffusion-expert
init, `encode_exact_durations` [17, 61, 73], 45056 packed tokens) and `compile.enabled=False` (required on 48 GB Ampere).
theta (full mode) = the `action_policy_libero_nano` trainable set: moe_gen + time_embedder + vae2llm + llm2vae + heads
(no `k_norm_und_for_gen` on Nano); LoRA targets add `mlp_moe_gen.gate_proj`.

| TOML | topology | inner batch | notes |
| --- | --- | --- | --- |
| `action_reptile_meta_nano.toml` (recommended) | shard 8 x 32 windows | 256 = K16 x w16, one pass/step | same optimisation as `action_reptile_meta_edge2.toml` (2 x 128); ~30 GB/GPU, ~3 min/iter, 1000 iters ~ 2 days |
| `action_reptile_meta_lora_nano.toml` | shard 8 x 32 | 256 | LoRA mode, lr 1e-4 / wd 0 |
| `action_reptile_meta_nano_smoke.toml` | shard 2 x 4 | tiny | 2 iterations, theta without moe_gen, no DCP (`save_at_end=false`) |
| `action_reptile_meta_nano_h200.toml` (8 x H200, downstream gbs 1024) | shard 8 x 128 | 1024 = K64 x w16, one full-batch step | k = 20 (20 epochs), warmup 2, lr 5e-5, q 16; ~50 GB/GPU, ~3 min/iter, 1000 iters ~ 2-2.5 days; downstream pair `action_policy_libero_10_nano_reptileinit_h200.toml` (8 x 128) |

Memory model (per GPU, shard 8, full activation checkpointing): fp32 master weights 63 GB + grads 28 GB + Adam 56 GB +
theta copy 28 GB = 175 GB / 8 = ~22 GB persistent, + ~0.22 GB per 256-res window (from the hs08 Nano post-training:
33 GB at 32 windows with an EMA copy instead of the theta copy). Compute: ~4.9x the Edge FLOPs per window -> ~15 s per
inner step at 32 windows/rank. A full-mode Nano DCP is ~120 GB (63 GB weights + materialised Adam moments), hence
`save_iter 500`. `[custom.meta] save_at_end=false` skips the final save (smoke runs).

```bash
export ROBOT_FEWSHOT_ROOT=$COSMOS_STORAGE/data/robot_fewshot BASE_CHECKPOINT_PATH=$COSMOS_STORAGE/checkpoints/Cosmos3-Nano
NPROC_PER_NODE=8 sr 8 48 examples/launch_reptile_meta_nano.sh
# downstream (few-shot Nano recipe = cosmos_hs08 action_policy_libero_10_nano_v2: shard 8 x 64, lr 5e-5, val 200 x 16)
JOB=$COSMOS_STORAGE/outputs/cosmos3_action_meta/reptile_meta/nano_reptile_full_k16w16_s10_eps0.5_seed42
export LIBERO_ROOT=... REPTILE_CKPT_PATH=$JOB/checkpoints/iter_000001000 META_ACTION_INIT_PATH=$JOB/meta_action_init_iter_001000.pt
NPROC_PER_NODE=8 sr 8 48 examples/launch_sft_action_policy_libero_10_nano_reptileinit.sh
# baseline with the same recipe and fresh heads: action_policy_libero_10_nano_fewshot.toml (launch with TOML_FILE=... via
# launch_sft_action_policy_libero_10_nano.sh or the hs08 launcher)
```

The few-shot Nano post-training experiment (`action_policy_libero_nano.py`) was taken from cosmos_hs08 (adds the fixed
episode subset, `dataloader_val` and the val callback; all off by default, so the original full-data
`action_policy_libero_10_nano.toml` is unchanged).

**v6 / v7 (2026-10-02)** = v2 + noise-level gate: the DINOv2 term is applied only to samples with `sigma <= 0.5` (v6,
low-noise / high-frequency half) or `sigma >= 0.5` (v7, high-noise / low-frequency half); `[model.repa].sigma_max` /
`sigma_min`, launchers `..._repa_dinov2_v{6,7}.sh` (ports 50055 / 50056). See docs/action_policy_libero_repa_vjepa.md section 10.

**v10 / v11 / v12 (2026-10-02)** = v2 with the DINOv2 teacher at (near) native size: v10 `target_subgrid_thw=[4,3,3]` @224 px,
v11 the same @210 px (exact, no pooling), v12 `student_upsampler="trilinear"` (student grid upsampled to the full 224 px
grid). Micro-batch 64x2 / 64x2 / 32x4 for memory. See docs/action_policy_libero_repa_vjepa.md section 11.

**v13 / v14 / v15 (2026-10-02)** = v2 / v10 / v12 with VideoREPA token-relation distillation instead of the token cosine
(`loss_weight 0`, `relation_loss_weight 5.0`, l1, `relation_margin 0.1`; full spatial+temporal similarity map per window).
Launchers `..._repa_dinov2_v1{3,4,5}.sh` (ports 50060-50062). See docs/action_policy_libero_repa_vjepa.md section 12.

**Nano post-training + DINOv2 REPA (2026-09-30)**: `action_policy_libero_10_nano_reptileinit_v2.toml` (launcher
`..._nano_reptileinit_v2.sh`, port 50050) = the plain Nano Reptile-init recipe + `[model.repa]` with the frozen DINOv2
ViT-L/14 teacher (`dinov2_vitl14`, 224 px, D=1024) on the LAST MoT block (`layer_index = 36`), plain token cosine, weight
5.0, projector 4096-2048-2048-1024. It needs experiment `action_policy_libero_nano_repa` (ported from cosmos_hs10: adds
`repa_` to `keys_to_select` / `keys_to_skip_loading` and ships the native frames for the teacher); the Reptile warm start
is unchanged (`[checkpoint]` fields). `repa_` is also in the TOML's `keys_to_skip_loading` because the Nano Reptile DCP
has no `repa_head`. The user's v2 / v3 / v4 = the same recipe at loss_weight 1.0 / 5.0 / 10.0. **Nano v5**
(`action_policy_libero_10_nano_reptileinit_v5.toml`, launcher `..._nano_reptileinit_v5.sh`, port 50051) = v2 +
`target_subgrid_thw = [1, 2, 2]` (less pooling, the Nano twin of the Edge v5 recipe; projector output 4 x 1024).

### 9b. Scaling the inner loop on big GPUs (H200): which knob buys what

The per-rank batch of a Reptile inner step is the support share per rank, not a free "bigger batch" knob: with a
full-batch inner loop, doubling it just halves the number of (smoother) steps. What the meta-init needs is an inner
loop that resembles the downstream post-training (30 demos, minibatch 256, many epochs, lr 5e-5), so spend compute as:

| lever | effect | recommendation |
| --- | --- | --- |
| `lr` | Adam moves ~lr per coordinate per step whatever the batch; Cosmos3 keeps 5e-5 from gbs 256 to 2048 | keep 5e-5 (the sqrt rule is about minibatch noise, irrelevant for a full-batch inner loop) |
| `k_shot` | demos per episode; bridge/fractal demos have only 20-28 windows so K, not `windows_per_demo`, grows the support | support = downstream gbs: K = gbs / 16 (64 at gbs 1024); 32 at gbs 512 |
| `windows_per_demo` | windows sampled per demo (repeats when a demo is shorter) | 16 |
| `max_samples_per_batch` x ranks | windows per inner step | = the downstream global batch (256 here, 1024 on the H200 plan); support / that = minibatches per epoch |
| `inner_steps` | epochs of adaptation per episode; the k=10 full-batch loop on 8 demos overfit (negative query gain) | 20 epochs over the support (k = 20 full-batch at gbs 1024, or 40 with 2 minibatches); warmup 10% |
| `q_query` | diagnostics only | 16 |
| `meta_batch_embodiments` | averages displacements before the meta step (smoother theta, x cost) | 1; 2 if time allows |
| `max_iter` | eps anneals over it; more iterations = more theta drift budget | 1000; 1500-2000 affordable on H200 |
| `compile.enabled` | works on Hopper (228 KB smem) and in principle with in-place theta rewrites, but untested in this trainer | leave off |


## 10. LIBERO-Goal / Object / Spatial suite variants of the reptileinit v3 recipe (2026-10-06)

`action_policy_libero_{goal,object,spatial}_edge_reptileinit_v3.toml` are byte-for-byte the LIBERO-10 v3 recipe except for
`[job].group/name` (`edge_libero_<suite>_fewshot` / `edge_libero_<suite>_3ep_fullft_reptile_lrx2`) and the two episode-subset
jsons. Nothing else in the code is suite specific: the dataset reads tasks / episodes / video paths from the suite's own `meta/`,
and the bundled `quantile_rot` stats were computed on all four suites pooled (see `normalizer_stats/*libero*.json` metadata),
so the same stats file serves training and the eval server.

Fixed subsets (`make_libero_episode_subset`, seed 42, same procedure as the LIBERO-10 files; never regenerate):

| suite | train json (3 demos/task = 30 eps) | val json (5 held-out demos/task = 50 eps, disjoint) | train windows |
|---|---|---|---|
| libero_goal (428 eps) | `libero_goal_3ep_per_task_seed42.json` | `libero_goal_val_5ep_per_task_seed42_excl3ep.json` | 3386 |
| libero_object (454 eps) | `libero_object_3ep_per_task_seed42.json` | `libero_object_val_5ep_per_task_seed42_excl3ep.json` | 3902 |
| libero_spatial (432 eps) | `libero_spatial_3ep_per_task_seed42.json` | `libero_spatial_val_5ep_per_task_seed42_excl3ep.json` | 3357 |

The loader validates every listed episode against the suite's episode -> task mapping, so a LIBERO-10 json used with another
suite's `LIBERO_ROOT` (or vice versa) fails at construction instead of training on the wrong demos. Run with the LIBERO-10
launcher and two overrides (the launcher pins `MASTER_PORT=50014`; give concurrent suite jobs on one node different ports):

```bash
LIBERO_ROOT=$COSMOS_STORAGE/data/LIBERO_LeRobot_v3/libero_goal \
TOML_FILE=examples/toml/sft_config/action_policy_libero_goal_edge_reptileinit_v3.toml \
NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_libero_10_edge_reptileinit_v3.sh
# closed-loop eval: the suite is TASK_SUITE (default libero_10 in examples/eval_libero_closed_loop.sh, which forwards it as
# closed_loop_eval.py --task_suite; max steps per suite come from TASK_MAX_STEPS there)
TASK_SUITE=libero_goal RUN_ROOT=<run dir> ITER=2000 OUT_DIR=<out> sr 1 48 examples/eval_libero_closed_loop.sh
```
