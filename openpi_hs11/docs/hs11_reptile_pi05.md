# cosmos_hs11 → openpi: Reptile meta pre-training between `pi05_base` and LIBERO fine-tuning

This fork adds to openpi (upstream `215abfb`) the cosmos_hs11 pipeline on top of π0.5:

```
pi05_base ──(Reptile over 5 robot embodiments: scripts/train_reptile.py)──► theta ──(LIBERO post-training: scripts/train.py)──► policy
                                                                   baseline: pi05_base ──(LIBERO post-training)──► policy
```

Everything lives under `src/openpi/hs11/`, `scripts/train_reptile.py`, `scripts/hs11/`, `examples/libero_v3/` and
`assets/episode_subsets/`. Upstream files touched: `training/config.py` (one `DataConfig.dataset_builder` field +
registration of the hs11 configs) and `training/data_loader.py` (honour `dataset_builder`).

## 0. Environment (done once)

```bash
cd ~/project/openpi_hs11
pip install --user uv                                   # ~/.local/bin/uv
export UV_CACHE_DIR=$COSMOS_STORAGE/uv_cache UV_PYTHON_INSTALL_DIR=$COSMOS_STORAGE/uv_python
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11             # -> .venv (jax 0.5.3 cuda12, torch 2.7.1, PyAV 17, pyarrow 20)
bash scripts/hs11/download_pi05_base.sh                 # gs://openpi-assets/checkpoints/pi05_base/params -> $COSMOS_STORAGE/openpi_cache (11.6 GB, done)
~/anaconda3/envs/libero-eval/bin/pip install msgpack websockets dm-tree   # eval client deps (done)
```

All launchers source `scripts/hs11/_common.sh`, which sets `OPENPI_DATA_HOME=$COSMOS_STORAGE/openpi_cache`,
`LIBERO_V3_ROOT=$COSMOS_STORAGE/data/LIBERO_LeRobot_v3`, `ROBOT_FEWSHOT_ROOT=$COSMOS_STORAGE/data/robot_fewshot`,
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` and `WANDB_MODE=offline`, and run `.venv/bin/python` (no `uv run` needed).
Use the site SLURM wrapper exactly as for cosmos_hs11 (`sr <gpus> 48 <script>`); the scripts detect the visible GPUs
and shard the model over all of them (`--fsdp-devices`).

## 1. Data: LeRobot v3.0 everywhere

openpi pins LeRobot at a pre-v3 commit, which cannot open the NVIDIA `LIBERO_LeRobot_v3` suites or the five
`robot_fewshot` embodiments (all LeRobot **v3.0**: `meta/episodes/chunk-*/file-*.parquet`, many episodes per data /
video file, `from_timestamp` offsets, AV1). `src/openpi/hs11/lerobot_v3.py` reads them directly (pyarrow + PyAV, no
`lerobot` import) and plugs into openpi through `DataConfig.dataset_builder`. Samples carry openpi keys, action chunks
are `rows[t:t+H]` padded with the last action at the episode end (LeRobot semantics), frames are decoded at their real
timestamps.

### LIBERO_LeRobot_v3 vs. what openpi's LIBERO policy expects

| | openpi `physical-intelligence/libero` | NVIDIA LIBERO_LeRobot_v3 | handled by |
|---|---|---|---|
| state | eef xyz, axis-angle, gripper qpos(2) = 8 | `observation.state` [8], same layout | identical |
| images | upright (rotated 180° from the sim render), 256² | `observation.images.{image,wrist_image}` 256², upright | identical; client rotates the render |
| actions | OSC command, gripper ∈ {−1, +1} (−1 = open) | `action` [7], gripper ∈ [0, 1] (**1 = open**) | train on raw v3; eval maps gripper `-1 if g>0.5 else +1` |
| rate | 10 fps metadata, 20 Hz sim | 20 fps | consistent: 10-step chunk = 0.5 s, replan every 5 steps |
| prompt | task text | `task_index` → `meta/tasks.parquet` text | `prompt` from the dataset |

`src/openpi/hs11/libero_v3.py` holds the spec, `LiberoV3Inputs/Outputs` (same slots as `LiberoInputs`) and
`v3_action_to_env`. The eval client `examples/libero_v3/main_v3.py` builds the observation exactly like
`examples/libero/main.py` and applies the gripper mapping.

### Episode subsets

`assets/episode_subsets/` is a verbatim copy of the cosmos_hs11 jsons. The recipes use the FINAL subsets:

| suite | 3 demos / task | 1 demo / task |
|---|---|---|
| libero_10 | `libero_10_3ep_per_task_seed42.json` | `libero_10_1ep_per_task_v1.json` |
| libero_goal | `libero_goal_3ep_per_task_v2.json` | `libero_goal_1ep_per_task_v1.json` |
| libero_object | `libero_object_3ep_per_task_seed42.json` | `libero_object_1ep_per_task_v1.json` |
| libero_spatial | `libero_spatial_3ep_per_task_v3.json` | `libero_spatial_1ep_per_task_v1.json` |

The reader validates every listed episode against the suite's episode→task mapping (wrong suite ⇒ error at
construction). "all" = every episode of the suite (openpi has no validation loop; judge by closed-loop evaluation).

### Reptile embodiments (`src/openpi/hs11/embodiments.py`)

| name | root | base camera | wrists | state | action (native, absolute for joints) | fps |
|---|---|---|---|---|---|---|
| fractal | google_robot_rt1 | image 256×320 | – | observation.state [8] | action [7] | 3 |
| bridge | bridge_v2 | image_0 480×640 | – | observation.state [7] | action [7] | 5 |
| robomind_ur | robomind/ur_1rgb | camera_top | – | states.joint_position [7] | actions.joint_position [7] | 30 |
| robomind_franka | robomind/franka_1rgb | camera_top 720×1280 | – | states.joint_position [8] | actions.joint_position [8] | 30 |
| molmoact2_yam | yam/repos (100 repos, `yam_100_repos_seed42.txt`) | top | left, right | observation.state [14] | action [14] | 30 |

Missing wrist views are zero images with `image_mask=False` (openpi's convention); states/actions are zero-padded to
32 dims and quantile-normalized with per-embodiment statistics.

## 2. Normalization statistics (CPU, once)

```bash
bash scripts/hs11/compute_norm_stats_libero_v3.sh            # assets/pi05_libero_v3_<suite>_all_base/<suite>/norm_stats.json (all 4 suites)
bash scripts/hs11/compute_norm_stats_reptile.sh              # assets/pi05_reptile/<embodiment>/norm_stats.json (200 k frames each)
```

Each LIBERO suite has ONE stats file (computed on all its episodes) shared by its six recipes (all/3ep/1ep × base/reptile)
through `AssetsConfig(assets_dir=./assets/pi05_libero_v3_<suite>_all_base)`; checkpoints copy it to
`<ckpt>/assets/<suite>/` and the policy server reads it from there.

## 2b. TOML-driven runs (the cosmos_hs11 way; recommended)

Every experiment is one TOML under `scripts/hs11/toml/` plus a launcher of the same name under `scripts/hs11/`.
The TOML re-states every run setting (batch, steps, schedule, EMA, wandb, init checkpoint, data subset); the
launcher only sources `_common.sh` and runs `scripts/train_toml.py --toml <file>` (or `train_reptile_toml.py`).
Any key can be overridden on the command line as `section.key=value`; unknown sections / keys are errors.

| experiment | TOML / launcher stem | base recipe |
| --- | --- | --- |
| Reptile pre-training (full theta, cosmos v4 defaults) | `reptile_pi05_v1` | `ReptileConfig` |
| k=1 joint-SFT control | `reptile_pi05_joint_v1` | `ReptileConfig` |
| LIBERO baselines (pi05_base init) | `{libero10,goal,object,spatial}_{all,3ep,1ep}_base_v1` | `pi05_libero_v3_<suite>_<data>_base` |
| LIBERO with Reptile init | `{libero10,goal,object,spatial}_{all,3ep,1ep}_reptile_v1` | `pi05_libero_v3_<suite>_<data>_reptile` |

```bash
# dry run on CPU: resolves the TOML, builds the dataset, prints every field
JAX_PLATFORMS=cpu .venv/bin/python scripts/train_toml.py --toml scripts/hs11/toml/goal_3ep_base_v1.toml --dryrun

# Reptile (4 GPUs) and its control
sr 4 48 scripts/hs11/launch_reptile_pi05_v1.sh
sr 4 48 scripts/hs11/launch_reptile_pi05_joint_v1.sh

# LIBERO baselines / Reptile init (edit [checkpoint].init_params in the *_reptile_v1 TOMLs or pass REPTILE_PARAMS)
sr 4 48 scripts/hs11/launch_goal_3ep_base_v1.sh
REPTILE_PARAMS=checkpoints/pi05_reptile/reptile_v1/1000/params sr 4 48 scripts/hs11/launch_goal_3ep_reptile_v1.sh

# overrides (GPU smoke, wandb off, different schedule)
sr 4 48 scripts/hs11/launch_goal_3ep_base_v1.sh train.num_train_steps=20 train.save_interval=10 wandb.enabled=false
sr 4 48 scripts/hs11/launch_goal_3ep_base_v1.sh job.exp_name=goal_3ep_base_lr2e5 optimizer.peak_lr=2e-5
```

`[wandb]`: `enabled` switches logging on/off, `mode` sets `WANDB_MODE` (offline by default; `wandb sync` later),
`dir` / `entity` are optional. `[train].fsdp_devices = 0` shards over every GPU of the allocation. New variants: copy a
TOML, change `exp_name` (checkpoint dirs are `checkpoints/<base recipe>/<exp_name>/`), launch with `TOML_FILE=...` or
a copied launcher. Loader: `src/openpi/hs11/toml_config.py` (tests: `toml_config_test.py`).

## 3. Reptile meta-training (`scripts/train_reptile.py`)

Full-parameter theta (all `nnx.Param` of π0.5 = PaliGemma + action expert + projections, ~3.3 B), JAX FSDP. Defaults =
`action_reptile_meta_edge_v4.toml`:

| cosmos_hs11 `[custom.meta]` | here (`ReptileConfig`) | value |
|---|---|---|
| k_shot / windows_per_demo / q_query | `--sampler.k-shot / --sampler.windows-per-demo / --sampler.q-query` | 32 / 8 / 8 (256 support, 64 query) |
| inner_steps, inner_warmup_steps, inner_grad_clip | `--inner-steps --inner-warmup-steps --inner-clip` | 10, 2, 1.0 |
| lr / weight_decay (= downstream) | `--inner-lr --inner-weight-decay` | 5e-5 / 1e-10 (openpi AdamW default) |
| inner_reset_optimizer | `--inner-reset-optimizer / --no-inner-reset-optimizer` | True |
| meta_lr, meta_lr_min_ratio, max_iter | `--meta-lr --meta-lr-min-ratio --max-iter` | 0.5 → 0.05 linear, 1000 |
| max_samples_per_batch (inner micro-batch) | `--inner-batch-size` | 32 (8 micro-batches, gradients averaged = ONE full-support step) |
| embodiment sampling | uniform over the five | – |
| disjoint_tasks (query) | `--sampler.disjoint-query-tasks` | False |

```bash
EXP_NAME=reptile_v1 sr 4 48 scripts/hs11/launch_reptile_pi05.sh                       # 4 GPUs, FSDP 4
EXP_NAME=reptile_joint sr 4 48 scripts/hs11/launch_reptile_pi05.sh --inner-steps 1 --meta-lr 1.0 --meta-lr-min-ratio 1.0 --no-inner-reset-optimizer   # k=1 control
EXP_NAME=reptile_v1 sr 4 48 scripts/hs11/launch_reptile_pi05.sh --resume               # (TOML launchers resume automatically)
```

Checkpoints: `checkpoints/pi05_reptile/<EXP_NAME>/<meta-iter>/params/` every `--save-interval` (250) meta-iterations
(standard openpi `params` directories, ~13 GB each), `reptile_train_log.jsonl` with per-iteration `support_loss_first/
last`, `query_loss_zero_shot/adapted` (every 5 iters), `adapt_delta`, `meta_step_norm`, `eps`, timings. The model
config is the downstream one (`pi05`, horizon 10, state not fed as discrete tokens) so the inner loop mirrors
post-training. Memory: fp32 theta + params + grad accumulator + Adam ≈ 65 GB sharded over the GPUs, plus activations
for 32/N samples per GPU; use ≥ 4 × 48 GB (or reduce `--inner-batch-size`).

## 4. LIBERO post-training (`scripts/train.py`, configs `pi05_libero_v3_<suite>_<data>_<init>`)

`suite ∈ {libero_10, libero_goal, libero_object, libero_spatial}`, `data ∈ {all, 3ep, 1ep}`, `init ∈ {base, reptile}`
(24 configs). All share the `pi05_libero` recipe: AdamW clip 1.0, EMA 0.999, quantile norm, global batch 64;
`all`: 30 k steps, 10 k warmup then constant 5e-5 (openpi's schedule); `3ep`: 5 k steps (200 warmup, cosine 5e-5 → 5e-6,
≈ 94 epochs of the 30 demos); `1ep`: 3 k steps. The Reptile variant only swaps the initial weights
(`$HS11_REPTILE_PARAMS`, set by the launcher from `REPTILE_PARAMS`).

```bash
# baseline (pi05_base) and Reptile init, 3 demos/task, LIBERO-goal, 2 GPUs
EXP_NAME=goal_3ep_base_v1 sr 2 48 scripts/hs11/launch_sft_libero_v3.sh libero_goal 3ep base
EXP_NAME=goal_3ep_reptile_v1 REPTILE_PARAMS=checkpoints/pi05_reptile/reptile_v1/1000/params \
    sr 2 48 scripts/hs11/launch_sft_libero_v3.sh libero_goal 3ep reptile
# all data
EXP_NAME=goal_all_base_v1 sr 4 48 scripts/hs11/launch_sft_libero_v3.sh libero_goal all base
# any TrainConfig field can be overridden, e.g. --num-train-steps 2000 --batch-size 32 --save-interval 500
```

Checkpoints: `checkpoints/<config>/<EXP_NAME>/<step>/{params,assets,train_state}`.

**Re-launching the same TOML (same `exp_name`) continues the run.** `[train].resume` / `[infra].resume` default to
`true`: if the checkpoint directory exists, training restores its latest `train_state` and keeps logging to the wandb
run whose id is stored in `<ckpt>/wandb_id.txt` (`wandb.init(id=..., resume="allow")`; offline mode keeps the id and
`wandb sync --append <dir>` merges the parts). A directory that exists without any saved step (crashed before the
first `save_interval`) starts over in place with a new wandb run. `scripts/train_toml.py` prints the decision
(`checkpoint dir ... -> RESUMING from step N` / `fresh start`) before training. `train.overwrite=true` wipes the
directory instead; `train.resume=false` restores openpi's refusal to touch an existing directory.

Checkpoint naming: openpi saves at loop index `step`, so with `num_train_steps = 5000` the directories are `1000, 2000, 3000, 4000, 4999`;
`4999` is the FINAL checkpoint (all 5000 updates applied; the saved `train_state.step` is 5000). `keep_period` keeps the
multiples, `max_to_keep=1` keeps the latest. Evaluate `.../4999` (or `.../29999` for the 30 k recipes, `.../2999` for 1ep).

## 5. Evaluation (`scripts/hs11/eval_libero_v3.sh`)

Starts `scripts/serve_policy.py` (openpi env, 1 GPU) on the checkpoint and runs `examples/libero_v3/main_v3.py` in the
`libero-eval` env with `~/project/LIBERO` + `packages/openpi-client/src` on `PYTHONPATH`:

```bash
# TOML-driven (the server rebuilds the exact training pipeline, incl. [model] changes, via scripts/hs11/serve_policy_toml.py)
TOML_FILE=scripts/hs11/toml/libero10_3ep_base_v1.toml CKPT=checkpoints/pi05_libero_v3_libero_10_3ep_base/libero10_3ep_base_v1/4999 \
OUT_DIR=$COSMOS_STORAGE/outputs/eval_openpi/libero10_3ep_base_v1_4999 NUM_TRIALS=50 SAVE_GIFS=1 sr 1 48 scripts/hs11/eval_libero_v3.sh
# or by registered config name
CONFIG=pi05_libero_v3_libero_goal_3ep_base CKPT=checkpoints/pi05_libero_v3_libero_goal_3ep_base/goal_3ep_base_v1/4999 \
OUT_DIR=$COSMOS_STORAGE/outputs/eval_openpi/goal_3ep_base_v1_4999 sr 1 48 scripts/hs11/eval_libero_v3.sh
```

Output `summary.json` has the cosmos_hs11 layout (`task_results[].episode_results[]`, `overall_success_rate`), so the
existing analysis snippets work. Max steps per suite and the 10-step wait are openpi's; `--replan_steps 5` executes
the first 5 of the 10 predicted actions (set `REPLAN_STEPS=10` for the cosmos-style full chunk). The gripper mapping is
binarized at 0.5 (`--gripper continuous` for `-(2g-1)`).

**Rendering / GPU selection.** MuJoCo renders through EGL. Inside a SLURM allocation the EGL device list keeps the
physical GPU order while CUDA renumbers the allocated GPU to `0`, and only the allocated GPU's `/dev/nvidia*` can be
opened (cgroup), so the client probes the list (`select_egl_device`): NVIDIA devices at the index SLURM reports
(`SLURM_STEP_GPUS`) first, other NVIDIA devices, then Mesa's software rasterizer last. The chosen index is exported as
`MUJOCO_EGL_DEVICE_ID` (and `CUDA_VISIBLE_DEVICES` is aligned to it in the client process only, for robosuite's import
assertion). `client.log` prints the device table and a `GL renderer:` line, the ground truth: `NVIDIA ...` = GPU,
`llvmpipe` = CPU. CPU rendering works but is ~5x slower (LIBERO-10 episode ~100 s instead of ~20 s), so the client
refuses it unless `ALLOW_CPU_RENDER=1` (`--allow_cpu_render`). To inspect a node in seconds:
`sr 1 48 ~/anaconda3/envs/libero-eval/bin/python examples/libero_v3/egl_probe.py` (device table + fps of a toy scene).

**Running faster.** 500 episodes take ~2.5 h on one GPU. Split the tasks over several allocations (each has its own
GPU, `PORT` and `OUT_DIR`, lab rule: only the allocated GPU is touched) and merge:

```bash
BASE=$COSMOS_STORAGE/outputs/eval_openpi/libero10_3ep_base_v1_4999
... OUT_DIR=$BASE/t0-4 PORT=8100 sr 1 48 scripts/hs11/eval_libero_v3.sh --task_ids 0,1,2,3,4
... OUT_DIR=$BASE/t5-9 PORT=8101 sr 1 48 scripts/hs11/eval_libero_v3.sh --task_ids 5,6,7,8,9
python scripts/hs11/merge_eval_summaries.py --out $BASE/summary.json $BASE/t0-4/summary.json $BASE/t5-9/summary.json
```

## 5b. Classifier-free guidance at evaluation (`GUIDANCE`)

`Pi0.sample_actions` (cosmos_hs11 addition) accepts `guidance` + an unconditional prompt: at every Euler step
`v = v_uncond + guidance * (v_cond - v_uncond)`, where the unconditional branch is the same observation with the prompt
replaced (default: the empty prompt). Both branches run as one batch of 2, so inference costs ~2x.

```bash
GUIDANCE=3.0 TOML_FILE=scripts/hs11/toml/libero10_3ep_base_v1.toml CKPT=.../4999 \
OUT_DIR=$COSMOS_STORAGE/outputs/eval_openpi/libero10_3ep_base_v1_4999_g3 sr 1 48 scripts/hs11/eval_libero_v3.sh
# UNCOND_PROMPT="..." changes the unconditional prompt (default ""); summary.json records guidance/uncond_prompt in server_metadata.
```

Caveat: unlike Cosmos3 (10 % caption dropout), openpi's pi05 recipes train without prompt dropout, so the "unconditional"
branch is an empty-prompt extrapolation rather than a trained unconditional model. To train a proper unconditional branch
set `[data].prompt_dropout = 0.1` in the training TOML (`libero_v3.PromptDropout`, dataset side only), then sweep
`GUIDANCE` on that checkpoint. `GUIDANCE` needs the TOML path (`TOML_FILE=`); the stock `scripts/serve_policy.py` has no CFG.

## 5c. Language ranking loss (`[model.language_ranking]`, cosmos_hs11 v23/v41 port)

Same objective as Cosmos' `algorithm/loss/language_ranking.py`: every training sample is predicted a second time from the
*same* noised chunk `x_t` / flow time under a uniformly drawn instruction of **another** task of the suite, and the
per-sample hinge `relu(margin + E_pos - E_neg)` (E = fp32 MSE over the chunk and the real action dims) is averaged over the
batch and added as `loss_weight * hinge` to the flow-matching loss. Samples whose prompt was replaced by `PromptDropout`
are flagged invalid and contribute zero. Pieces: `pi0_config.LanguageRankingConfig` (knobs: `enabled`, `loss_weight`,
`margin`, `raw_action_dim`, `negative_grad_scale`, `margin_mode`), `libero_v3.NegativePrompt` (repack side, after the
dropout; negatives come from `meta/tasks.parquet` of the whole suite), `transforms.TokenizeNegativePrompt` (model side,
no-op without a negative), `Observation.tokenized_negative_prompt(_mask)` / `ranking_valid`, and
`Pi0.compute_loss_and_metrics` (used by `train.py`, logs `language_ranking_{loss,weighted_loss,positive_error,
negative_error,gap,margin_satisfied,valid_fraction}`). Cost: a second prefix+suffix forward/backward per step (~2x).

```toml
[model.language_ranking]
enabled        = true
loss_weight    = 0.022   # Cosmos: 1.0 next to a 10x action loss over the REAL dims -> 1:10 per element.
margin         = 0.1     # openpi's action loss averages over all 32 padded slots, so 0.1 * 7/32 = 0.022 is equivalent.
raw_action_dim = 7       # 0/omitted = infer the real dims from the batch (padding is exactly zero)
```

Tests: `src/openpi/hs11/language_ranking_test.py` (closed-form hinge on the dummy pi05, transforms, TOML knob);
CPU smoke: `scripts/hs11/smoke_train_cpu.py --language-ranking` (the dummy model's adaLN-zero init ignores the prompt, so
its gap is exactly 0 there; with `pi05_base` it is not).

## 6. Tests and CPU smokes

```bash
# unit / end-to-end tests with the dummy pi05 (~15 min on CPU; /tmp of the login node is full -> --basetemp on /gallery_moma)
JAX_PLATFORMS=cpu .venv/bin/python -m pytest --basetemp=$COSMOS_STORAGE/pytest_tmp -p no:cacheprovider \
    src/openpi/hs11/lerobot_v3_test.py src/openpi/hs11/configs_test.py scripts/train_reptile_test.py
# openpi's scripts/train.py on the REAL libero_goal 3-demo data with the dummy model (2 spawned workers, 3 steps, ~3 min)
JAX_PLATFORMS=cpu .venv/bin/python scripts/hs11/smoke_train_cpu.py --config pi05_libero_v3_libero_goal_3ep_base --out $COSMOS_STORAGE/smoke_openpi_hs11
```

Status 2026-10-13 (all CPU, nothing GPU-run):
* `lerobot_v3_test.py` (5), `configs_test.py` (4), `train_reptile_test.py` (3: 3 meta-iterations + checkpoint + resume +
  `CheckpointWeightLoader` round trip; 2 spawned workers) pass.
* `scripts/hs11/smoke_train_cpu.py`: `train.py` end to end on the real goal 3-demo subset (video decode in spawned
  workers, norm stats, train_step, checkpoint with `assets/libero_goal/norm_stats.json`) -> OK.
* served-policy path: `create_trained_policy(config, <ckpt>/<step>)` + `policy.infer(client observation)` -> (10, 7)
  actions, `v3_action_to_env` gripper in {-1, +1}, `policy_metadata` propagated -> OK.
* norm stats computed: the four LIBERO suites (`assets/pi05_libero_v3_<suite>_all_base/<suite>/`) and the five
  embodiments (`assets/pi05_reptile/<name>/`, 200 k frames each).
* pi05_base params downloaded to `$COSMOS_STORAGE/openpi_cache`.

Trap: never drive a spawning DataLoader from a stdin script (`python - <<EOF`): the spawned workers try to re-import
`__main__` from `<stdin>`, die, and the parent blocks forever in a pipe write (this cost a night of waiting).
Use real script files (as every launcher does).

## 7. Caveats

* Not GPU-smoked. First GPU runs to do: `EXP_NAME=smoke sr 4 48 scripts/hs11/launch_reptile_pi05.sh --max-iter 2 --save-interval 1`
  and `EXP_NAME=smoke sr 2 48 scripts/hs11/launch_sft_libero_v3.sh libero_goal 3ep base --num-train-steps 20 --save-interval 10`;
  expect the first step to compile for a few minutes (jax cache in `~/.cache/jax`).
* openpi's LIBERO recipe does not feed the proprioceptive state to π0.5 (`discrete_state_input=False`, no continuous
  state projection in pi05); kept for parity with `pi05_libero`.
* Native actions of the five embodiments are used as-is (RT-1/Bridge deltas, RoboMIND/YAM absolute joint targets).
* Frame decoding is per sample (seek + decode with PyAV, ~50-100 ms for the two LIBERO views); use `--num-workers 8+`
  on the training nodes. Expected speed: Reptile ~2-3 min per meta-iteration on 4 x 48 GB (80 forward/backward passes
  of 32 samples), LIBERO fine-tuning ~1-2 s per step of 64.
* The Reptile recipes read `$HS11_REPTILE_PARAMS` when `openpi.training.config` is imported; the launcher sets it from
  `REPTILE_PARAMS`. For the policy server pass the same variable (or `--policy.config` of the base twin, which has the
  identical data pipeline) when serving a Reptile-initialized checkpoint.
