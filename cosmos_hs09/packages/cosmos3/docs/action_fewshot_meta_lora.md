# Few-shot meta-training in the LoRA regime (cosmos_hs09)

`cosmos_hs09` = `cosmos_hs07` (see [action_fewshot_meta.md](./action_fewshot_meta.md) for the data readers,
episodic sampler and the FOMAML trainer) with one structural change: **theta_meta covers the whole
downstream trainable set**, and the downstream LIBERO-10 post-training is a **LoRA** fine-tune whose
trainable set is exactly theta_meta.

## 1. Why

In cosmos_hs07 the meta-learned initialization was the action heads only (266K parameters). The Cosmos
post-training recipe then fine-tuned `moe_gen` (1.4B) + `time_embedder` + `vae2llm` / `llm2vae` + the
heads at `lr 5e-5` (heads x5) for 2000 AdamW steps on 30 demonstrations. Two things followed and were
measured on LIBERO-10 (3 demos / task):

1. **Footprint**: the init differed on 0.02 % of what was being trained; the loss trajectory is set by the
   `moe_gen` fine-tune.
2. **Wash-out**: an Adam step moves every coordinate by ~lr regardless of the gradient, so after ~200
   steps (`sum lr_t ~ 0.01` at 5x) the head init is a small perturbation of a fresh init. Baseline and
   meta-init training curves coincided from iteration ~100-200 on, and closed-loop success was the same.

cosmos_hs09 fixes both: theta_meta = LoRA adapters on the generation tower + heads + `time_embedder`
(~38M params), and post-training optimizes exactly those parameters (LoRA regime, single LR 1e-4).
The init now covers 100 % of the trainable parameters and there is no large full-fine-tuned block to
dominate the trajectory. Configuration **A** (action-only meta loss) is used: `vae2llm` / `llm2vae`
stay frozen everywhere (`llm2vae` gets zero gradient from the action loss anyway).

## 2. What is in theta_meta

| group | parameters | # | meta init | downstream baseline init |
| --- | --- | --- | --- | --- |
| `action_heads` | `action2llm` / `llm2action` fc + bias rows (64-D padded) | 264,256 | Cosmos fresh init | Cosmos fresh init |
| `action_modality_embed` | shared vector | 2,048 | fresh | fresh |
| `lora` | `lora_A` / `lora_B` of `q/k/v/o_proj_moe_gen`, `mlp_moe_gen.up/down_proj`, 28 layers, rank 32, alpha 64 | 33,030,144 | A ~ kaiming, B = 0 (identity adapter) | same |
| `time_embedder` | `time_embedder.mlp.*` | 4,722,688 | mid-trained (base checkpoint) | mid-trained |

The heads live in a scratch `DomainAwareLinear` row (31) during meta-training and are copied into the
LIBERO row (5) downstream; the shared groups are written in place by name. Names are canonicalized
(`_checkpoint_wrapped_module.` / `_orig_mod.` stripped), so activation-checkpoint or compile wrappers on
either side do not matter. `meta_action_init.pt` files from cosmos_hs07 (heads only) still load; the
LoRA / `time_embedder` groups are then simply not written.

Infrastructure reused: `utils/generator/lora.py` (`LoraInjectedLinear` keeps the base `.weight` key, so
LoRA and non-LoRA checkpoints interoperate) and `model.config.lora_*`; wired in `OmniMoTModel.build_net`.

## 3. Files

| piece | path |
| --- | --- |
| meta adapter (groups, in-place fast weights, apply) | `cosmos_framework/data/generator/action/meta/meta_action_adapter.py` |
| meta trainer (`[custom.meta] include_lora / include_time_embedder`, `adapt_delta/<group>`) | `cosmos_framework/scripts/train_action_meta.py` |
| meta experiment | `configs/base/experiment/action/meta/action_fewshot_meta_lora_edge.py` |
| meta run / smoke TOML | `examples/toml/sft_config/action_fewshot_meta_lora_edge{,_smoke}.toml` |
| meta launcher | `examples/launch_meta_action_fewshot_lora_edge.sh` (MASTER_PORT 50015) |
| vision-loss variant (6c) | `examples/toml/sft_config/action_fewshot_meta_lora_edge_vision_loss.toml` + `examples/launch_meta_action_fewshot_lora_edge_vision_loss.sh` (MASTER_PORT 50017) |
| L2-SP variant (6d) | `examples/toml/sft_config/action_policy_libero_10_edge_metainit_l2sp.toml` + `examples/launch_sft_action_policy_libero_10_edge_metainit_l2sp.sh`; `callbacks/l2sp.py` |
| downstream LoRA recipe (train + **val** loaders) | `configs/base/experiment/action/posttrain_config/action_policy_libero_lora_edge.py` |
| baseline / ours TOML | `examples/toml/sft_config/action_policy_libero_10_lora_edge{,_metainit}.toml` |
| baseline / ours launcher | `examples/launch_sft_action_policy_libero_10_lora_edge{,_metainit}.sh` |
| held-out val subset (50 demos, disjoint) | `data/generator/action/episode_subsets/libero_10_val_5ep_per_task_seed42_excl3ep.json` |
| val loss callback | `cosmos_framework/callbacks/val_loss_breakdown.py` |
| meta-init injection (all groups) | `cosmos_framework/checkpoint/meta_action_init.py` |
| closed-loop eval | `examples/eval_libero_closed_loop.sh` (unchanged; builds the model from the run's `config.yaml`, which carries `lora_enabled`) |

Framework changes needed for this (all small):

* `model.config.lora_freeze_base` (default True = old behaviour). LoRA injection freezes every non-LoRA
  parameter, and the optimizer builder only considers parameters that already require grad -- so the
  heads / `time_embedder` listed in `keys_to_select` would have **silently** dropped out of the
  optimizer. The LoRA recipes set it to False and let `keys_to_select` decide.
* `OmniMoTModel.validation_step` was a stub; it now runs `training_step` under `no_grad` (under EMA
  when enabled). The trainer restores `model.train()` after a validation pass.
* TOML schema: `[trainer] run_validation / validation_iter / max_val_iter / run_validation_on_start`,
  `[dataloader_val] max_samples_per_batch / episode_subset_path`, `[model] lora_freeze_base`,
  `[checkpoint] meta_action_init_include_lora / _time_embedder`.
* `scripts/make_libero_episode_subset.py --exclude-subset` (disjoint validation subsets).

## 4. Hyper-parameters and why

**LoRA**: rank 32, alpha 64 (scale 2), targets q/k/v/o + mlp up/down of the `moe_gen` tower, dropout 0.
Attention-only (the `vision_sft_super` default) would be 13M params; including the MLP gives the
adapter a path through the tower's non-linear capacity, which is where the denoising actually happens.
Meta and downstream **must** use the same rank/alpha/targets (`apply_meta_action_init_to_net` checks shapes).

**Downstream LR: `1e-4` for everything, `lr_multipliers = {}`, `weight_decay = 0`.**
The 5x head multiplier of the Cosmos recipe exists because the heads were the one fresh module of a full
fine-tune and had to catch up with a pretrained 1.4B block. In the LoRA regime every trainable module is
adapter-sized, and for the meta-init arm a 5x LR only erases the init faster. Using one LR for both arms
keeps the comparison fair; 1e-4 is the conservative end of the usual LoRA range (30 demos ~ 220 epochs in
2000 steps, so overfitting is the concern, not underfitting). Weight decay 0 follows the codebase's LoRA
recipe; at 1e-4 x 2000 steps the 0.05 of the old recipe was a 1 % shrink anyway. Schedule (warmup 500,
linear cycle 16000) and batch (128 x 4 ranks) are unchanged from the Cosmos recipe.

**Meta**: inner Adam `1.2e-4` x 5 steps (per-coordinate step ~ lr, independent of the gradient scale --
in cosmos_hs07 SGD's adaptation shrank 0.24 -> 0.05 as the gradient shrank), outer AdamW `1e-4`, K = Q = 5
demonstrations x 8 windows (one 40-window batch each), action-only loss. Watch `adapt_delta/<group>`
(per-group adaptation magnitude) and `query_loss` vs `query_loss_zero_shot`; if the gap collapses raise
`inner_lr`.

## 5. Validation (`val/*` in wandb)

`dataloader_val` streams 50 held-out demos (5 / task, seed 42, disjoint from the 30 training demos;
`libero_10_val_5ep_per_task_seed42_excl3ep.json`) with no prompt dropout and a fixed shuffle seed; every
pass restarts at epoch 0 so successive numbers are computed on the same windows. Every
`validation_iter` (100) steps and **at iteration 0** (= the initialization itself, before any update),
`max_val_iter` (16) x 128 x 4 ranks ~ the whole set is evaluated under EMA and logged:

* `val/loss_total` (also `val/loss` from the stock `wandb_val` callback) -- vision x10 + action,
* `val/flow_matching_loss_action` -- **the number that matters for the policy**,
* `val/flow_matching_loss_vision`, `val/num_batches`, `val/num_nonfinite`.

Compare baseline vs ours on `val/flow_matching_loss_action` at iteration 0 (init quality) and along
training (generalization from 3 demos); training loss alone cannot separate fitting from generalizing.

## 6. Commands

```bash
cd ~/project/cosmos_hs09/packages/cosmos3
export ROBOT_FEWSHOT_ROOT=$COSMOS_STORAGE/data/robot_fewshot
export BASE_CHECKPOINT_PATH=$COSMOS_STORAGE/checkpoints/Cosmos3-Edge
export LIBERO_ROOT=$COSMOS_STORAGE/data/LIBERO_LeRobot_v3/libero_10

# (0) optional 1-GPU smokes
TOML_FILE=examples/toml/sft_config/action_fewshot_meta_lora_edge_smoke.toml NPROC_PER_NODE=1 sr 1 48 \
  examples/launch_meta_action_fewshot_lora_edge.sh

# (1) meta-training  -> $IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_meta/fewshot_meta_lora/edge_meta_lora_r32_fomaml_adam_k5q5_w8_seed42/
NPROC_PER_NODE=4 sr 4 48 examples/launch_meta_action_fewshot_lora_edge.sh

# (2) baseline: LoRA post-training from the mid-trained base
NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_libero_10_lora_edge.sh

# (3) ours: same recipe from theta_meta (an iteration snapshot, or meta_action_init.pt = latest)
export META_ACTION_INIT_PATH=$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_meta/fewshot_meta_lora/edge_meta_lora_r32_fomaml_adam_k5q5_w8_seed42/meta_action_init_iter_001500.pt
NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_libero_10_lora_edge_metainit.sh

# (4) closed-loop eval (either run)
RUN_ROOT=$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_libero/edge_libero10_fewshot_lora/<name> ITER=1500 OUT_DIR=$COSMOS_STORAGE/outputs/eval/<name>_it1500 \
  sr 1 48 examples/eval_libero_closed_loop.sh
```

What to check in the logs:

* meta: `MetaActionAdapter: ... groups={'action_heads': 264256, 'action_modality_embed': 2048, 'lora': 33030144, 'time_embedder': 4722688}`,
  `LoRA injection successful: 168 modules wrapped`.
* downstream: `lora_freeze_base=False: non-LoRA parameters left trainable`, a trainable-parameter count of
  ~38M (`param_count` callback), `[val] iter 0: ...` before the first step, and for ours
  `[meta-action-init] net: wrote 345 tensors {'action_heads': 4, 'action_modality_embed': 1, 'lora': 336, 'time_embedder': 4}`.

## 6b. Variant: meta init -> FULL fine-tune (original Cosmos recipe)

`examples/toml/sft_config/action_policy_libero_10_edge_metainit.toml` (launcher
`launch_sft_action_policy_libero_10_edge_metainit.sh`) post-trains with the **original** recipe
(`action_policy_libero_edge`: full `moe_gen` + `time_embedder` + `vae2llm` / `llm2vae` + `k_norm_und_for_gen` +
heads trainable) but starts from the LoRA-regime theta_meta:

* the model is built **with** the LoRA modules (same rank / alpha / targets as the meta run) so theta_meta's
  adapters can be copied in; they are then **frozen** with `[optimizer.lr_multipliers] lora_ = 0.0`, i.e. they
  act as a constant delta on `moe_gen` -- equivalent to merging the meta-learned LoRA into the base weights.
  (`"moe_gen"` in `keys_to_select` also matches the adapter names, hence the multiplier instead of exclusion;
  the experiment carries a `"lora_": 1.0` default entry only so a TOML can override it -- Hydra cannot add keys.)
* `lora_freeze_base = false` so the injector does not freeze `moe_gen`.
* action-head multiplier 5x -> **1x**: the 5x boost exists for *fresh* heads; here they are meta-initialized.
  This is a second difference to the fresh-head baseline besides the init -- run the baseline with 1x too if a
  single-variable comparison is needed.
* schedule mirrors the cosmos_hs08 full-FT baseline: 2 GPUs (shard 2), global batch 256, lr 5e-5, warmup 200,
  `cycle_lengths = [2000]`, 2000 iters; held-out validation as in section 5.
* `keys_to_skip_loading` must include `"lora_"` (the base DCP has no adapter tensors).

The eval server rebuilds the LoRA modules from the run's `config.yaml`, so the trained checkpoint evaluates
like any other.

## 6c. Variant: vision loss in the OUTER loop (`action_fewshot_meta_lora_edge_vision_loss.toml`)

The base recipe meta-trains with the action loss in both loops. Measured on LIBERO (`val/*` at iteration 0), that
theta_meta degraded the video term 3.2x (vision 0.61 vs 0.19 for the fresh baseline; action 2.88 vs 1.42) -- the
5-step FOMAML "launch pad" moves the shared trunk (LoRA on every `*_moe_gen` projection, the single `time_embedder`
used for vision AND action timesteps) off the video manifold of an objective it never saw. Both gaps close by
downstream iteration 100, so this is a second-order effect; the variant exists to test whether keeping the video
capability intact at theta_meta helps the first few hundred post-training steps.

Design: the INNER loop is unchanged (action-only, `inner_lr` keeps its meaning). The OUTER (query) objective becomes

    outer = raw flow_matching_loss_action + outer_vision_weight * raw flow_matching_loss_vision   (query demos)

i.e. "after adapting actions on the K support demos, the model should predict both actions and video well on the Q
held-out demos". The query video is already tokenized and forwarded in the base recipe (mode `wam`, `vision_gen=True`);
only the outer backward grows. New parameters are trained? No -- `llm2vae` (the only vision-exclusive tensor) is not in
this theta_meta; the term only redirects the shared-trunk gradient.

Why RAW terms and weight 1.0: the model stores `flow_matching_loss_vision/_action` unscaled and folds the x10 into
`total_loss` only (`omni_mot_model.py` around 1690/1767), so weight 1.0 reproduces Cosmos3's own 1:1 balance at the
scale the base run's `meta_lr = 1e-4` was tuned for. **Do not emulate this with `loss_mode = "total"`** -- that also
multiplies the action gradient by 10 (a confounded 10x step-size change). `from_dict` rejects
`outer_vision_weight > 0` together with `loss_mode = "total"` for that reason.

Data: the Q query demos of the sampled embodiment, nothing else. Do not add LIBERO or external video -- it leaks the
target domain into the meta stage and weakens the embodiment-agnostic claim.

Tuning: watch `query_loss_vision` (adapted, raw) and `query_loss_vision_zero_shot`; they should stay near the ~0.11 the
shared init produces on the meta embodiments instead of rising. `query_loss` still reports the raw ACTION term, so it is
directly comparable with the base run: if it ends > 20% above the base run's ~0.08, lower the weight to 0.3-0.5; if
vision still rises, raise it to 2-3. `outer_loss` = action + w x vision is logged as well.

Files: `examples/toml/sft_config/action_fewshot_meta_lora_edge_vision_loss.toml` (= base TOML + `outer_vision_weight`,
run name `edge_meta_lora_r32_fomaml_adam_k5q5_w8_vision1.0_seed42`), `examples/launch_meta_action_fewshot_lora_edge_vision_loss.sh`
(MASTER_PORT 50017), knob `MetaTrainConfig.outer_vision_weight` + `_outer_terms` in `scripts/train_action_meta.py`,
tests `scripts/train_action_meta_test.py`. With the default 0.0 the outer objective is the very same tensor as before
(tested), so the base launcher/TOML behave exactly as they did.

```bash
NPROC_PER_NODE=4 sr 4 48 examples/launch_meta_action_fewshot_lora_edge_vision_loss.sh
# downstream: unchanged -- META_ACTION_INIT_PATH=<...>/fewshot_meta_lora/edge_meta_lora_r32_fomaml_adam_k5q5_w8_vision1.0_seed42/meta_action_init.pt
```

## 6d. Variant: meta init -> full fine-tune with L2-SP toward the init (`action_policy_libero_10_edge_metainit_l2sp.toml`)

Purpose: in the full-FT recipe (6b) the meta-initialised groups -- head rows, `action_modality_embed`,
`time_embedder` -- are ordinary trainable parameters, and 2000 Adam steps can move each weight by up to
`sum(lr_t)` ~ 0.05, i.e. more than the ~0.02 init scale. L2-SP (Li, Grandvalet & Davoine, 2018) replaces the L2
pull toward zero by a pull toward the *starting point*; here it is applied **decoupled**, AdamW-style, after every
optimizer step (`callbacks/l2sp.py`):

    p <- p - lr_t * alpha * (p - p0)           lr_t = current lr of p's optimizer group (schedule + lr_multipliers)

Why decoupled and not a loss term: through Adam a loss penalty is rescaled per coordinate by the second-moment
estimate, so its strength is no longer `alpha`. Decoupled, the rule has a clean reading: under a persistent Adam drift
(~lr per step) the deviation saturates at **~1/alpha per weight, independent of lr**. alpha 100 caps a 0.02-scale meta
head weight at ~0.01 of sustained drift while non-persistent gradients move it far less; alpha 20 would allow 0.05 (= no
protection), alpha 1000 would nearly freeze the group. The e-folding time of a deviation at peak lr is 1/(lr*alpha) =
200 steps for lr 5e-5, alpha 100.

`p0` is captured at `on_train_start` of the fresh run, i.e. after checkpoint load *and* meta-init injection, and saved
per rank to `<job dir>/l2sp_anchors/rank{r}_of_{W}.pt`; a resume reloads those original anchors (same world size
required) instead of re-anchoring at the resumed weights. The pull runs on the local FSDP shards before the EMA update,
so `net_ema` follows.

Which groups: the TOML anchors exactly the meta-initialised trainable groups (`action2llm`, `llm2action`,
`action_modality_embed`, `time_embedder`) at alpha 100. The meta LoRA is frozen anyway (lr multiplier 0 -> rate 0). The
trunk (`moe_gen`, `vae2llm`, `llm2vae`, `k_norm_und_for_gen`) is left at 0: anchoring it toward the base checkpoint is the
classic L2-SP regulariser against overfitting 30 demos, but it only starts to bind around alpha 500-1000 (measured
full-FT trunk movement ~1e-3 per weight) and costs ~2.8 GB/GPU of fp32 anchors on 2 GPUs; turn it on deliberately.

Diagnostics (wandb + console, every `log_every` steps): `l2sp/rel_dev/<pattern>` = ||p - p0|| / ||p0|| over that
pattern's tensors (the 32-row head tables include 31 rows that never move, so their number is diluted; compare runs, not
absolute values) and `l2sp/rate/<pattern>` = lr_t * alpha. Tuning: if `val/flow_matching_loss_action` lags the plain
metainit run (6b) at the same iteration, alpha is too strong -> 30-50; if `rel_dev` of the meta groups still climbs like
the 6b run, raise to 200-300.

Plumbing: `[trainer.callbacks.l2sp]` in the TOML schema (`toml_config/sft_config.py`), the `l2sp` entry with all alphas 0
in the `action_policy_libero_edge` callbacks dict (a complete no-op for every other recipe), tests
`callbacks/l2sp_test.py`. Only the run name and the `[trainer.callbacks.l2sp]` section differ from the 6b TOML (tested).

```bash
export LIBERO_ROOT=... META_ACTION_INIT_PATH=<meta run>/meta_action_init.pt
NPROC_PER_NODE=2 sr 2 48 examples/launch_sft_action_policy_libero_10_edge_metainit_l2sp.sh
```

## 7. Caveats

* Meta uses the action-only loss; downstream keeps the Cosmos total loss (vision x10 + action) for
  both arms. If the vision term is suspected of dominating the LoRA trajectory, ablate with
  `EXTRA_TAIL_OVERRIDES="model.config.rectified_flow_training_config.loss_scale=0.0"` on both arms.
* Validation runs under EMA (`ema.enabled=true`) -- same weights the eval server sees.
* A checkpoint written by the LoRA recipe contains base + adapters (36 GB DCP as before); the eval
  server rebuilds the LoRA modules from the run's `config.yaml`.
* Single-node launches set `NCCL_IB_DISABLE=1` (node `bob` has a broken RoCE stack).
* Launching from a non-interactive shell (e.g. a plain `srun bash script.sh`) needs the conda env on
  `PATH` (`export PATH=$HOME/anaconda3/envs/cosmos3-pt/bin:$PATH`); `sr` from an activated shell is fine.

## 8. Cosmos3-Nano tier (`action_fewshot_meta_lora_nano`, 8 x H200)

`examples/toml/sft_config/action_fewshot_meta_lora_nano.toml` + `examples/launch_meta_action_fewshot_lora_nano.sh`
(port 50052, `NPROC_PER_NODE` default 8) run the same FOMAML LoRA meta-training on Cosmos3-Nano (Qwen3-VL-8B MoT, 36
blocks, hidden 4096, 15.75 B params). Experiment `action_fewshot_meta_lora_nano` = the Edge meta experiment on
`NANO_MODEL_CONFIG` with the Nano LIBERO-recipe deltas (`loss_scale` 10 / `image_loss_scale` None, `load_weights_from_pretrained`
False, `encode_exact_durations` [17, 61, 73], 45056 packed tokens), LoRA targets extended with the Qwen3 SwiGLU
`mlp_moe_gen.gate_proj`, and `action_pos_embed` (a Nano-DCP-only key) added to `keys_to_skip_loading`. The model is a
replicated bf16 copy without FSDP (31.5 GB per rank), so the recipe needs H200-class memory; a 48 GB card only fits
`action_fewshot_meta_lora_nano_smoke.toml` (1 GPU, 3 iterations, 4-window batches).

Sizing for a downstream Nano LoRA post-training of 8 GPUs x 128 windows (2048/step) for 2000 steps at lr 1e-4:
FOMAML cannot mirror a 2000-step adaptation, so the inner loop is made as downstream-like as memory allows --
`inner_lr = 1e-4` (the downstream lr; Adam moves each coordinate ~lr per step, so adaptation magnitude = steps x lr),
`inner_steps = 10` (Edge: 5), support `k_shot 8 x windows_per_demo 8 = 64` windows in one packed batch per inner step,
query `q_query 4 x 8 = 32` windows; `windows_per_demo 16` + `max_samples_per_batch 128` is the 128-window option (~2x
time). 8 ranks = 8 meta-episodes per outer step (the 4-GPU Edge run had 4). Estimated ~2.3 min/iter on H200 (Edge/A40:
0.135 s per window fwd+bwd, 0.43 s/window VAE encode; Nano ~4.6x params, H200 ~3.5x A40) -> 1000 iters ~38 h, 2000 ~77 h;
the outer LR anneals to `meta_lr_min_ratio` at `max_iter`, so set 1000 or 2000 rather than stopping early. The downstream
Nano LoRA recipe must use the same rank / alpha / `lora_target_modules` (with `lora_freeze_base = false`) and load the
meta init via `checkpoint.meta_action_init_path` + `meta_action_init_include_lora/_time_embedder`; no such Nano TOML exists
in this repo yet. Tests: `cosmos_framework/scripts/train_action_meta_nano_test.py`.
