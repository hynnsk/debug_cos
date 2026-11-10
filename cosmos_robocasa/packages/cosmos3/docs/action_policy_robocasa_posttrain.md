# RoboCasa (original, 24 tasks x ~50 human demos) post-training of Cosmos3-Edge (cosmos_robocasa)

Mid-trained Cosmos3-Edge (**baseinit**) vs. its Reptile meta-init (**reptileinit**, `docs/action_reptile_meta.md`)
post-trained on the ORIGINAL RoboCasa benchmark (RSS 2024) in the setting other papers use: all 24 atomic kitchen tasks,
the ~50 human teleoperation demos per task, one multi-task policy. The data and the recipe follow
**Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning** (Kim et al., ICLR 2026, arXiv 2601.16163,
github.com/NVlabs/cosmos-policy, 67.1 % average success with 50 demos/task) wherever the Cosmos3 action stack allows it.

| | Cosmos Policy (RoboCasa) | this recipe (Cosmos3-Edge) |
|---|---|---|
| data | `nvidia/RoboCasa-Cosmos-Policy` `success_only/`: 24 tasks, ~50 human demos/task replayed at 224x224, no-op actions filtered, success-trimmed, failed replays dropped | the same HDF5 files, converted once to LeRobot v3 shards (`examples/setup_robocasa.sh`) |
| action | first 7 of the 12D env action `[eef_pos(3), eef_rot_axisangle(3), gripper(1)]`, min/max -> [-1, 1]; chunk 32, execute 16 | the same 7D command -> 10D `[pos(3), rot6d(6), gripper(1)]` (`use_base_action=False`), `quantile_rot` (q01/q99 -> [-1, 1]) with the bundled `robocasa_cosmos_policy_*` stats; **chunk 16 / execute 16** (the LIBERO / Reptile window contract) |
| proprio | 9D `[gripper_qpos(2), eef_pos(3), eef_quat(4)]`, min/max -> [-1, 1], one latent frame | the same quantities as the 10D state token `[eef_pos, rot6d, finger opening]` prepended as a clean conditioning action (`use_state=True`), min/max-normalized by its own `state` statistics (`state_normalization="minmax"`) |
| observation | three 224x224 cameras (left, right, wrist), each its own latent frame | `camera_set="left_wrist"`: `agentview_left \| eye_in_hand` at 224x448 -> the 256-tier 192x320 bucket (LIBERO geometry). `wrist_lr` tiles all three (wrist full + left/right half size) if you want the third camera |
| augmentation | "stronger" aug: random crop 90 % area + resize, rotation +-5 deg, ColorJitter 0.3 / 0.4 / 0.5 / 0.05 | the same (train only; `[dataloader_train].use_image_augmentation`) |
| language | T5 embeddings, **no text dropout** | Qwen VLM prompt, `cfg_dropout_rate=0.0` -> evaluate with guidance 1.0 |
| optimizer | AdamW lr 1e-4, betas (0.9, 0.99), eps 1e-8, wd 0.1 (Predict2 defaults), full fine-tune, no EMA | the same values (FusedAdam, fp32 master weights), LIBERO Edge trainable set (= the Reptile theta); EMA kept ON (Cosmos3 eval servers load EMA weights; `[model.ema].enabled=false` to mirror them) |
| schedule | warmup 1000, linear decay to 0.3x over 30k of 45k steps, then 0.06x constant | the same shape scaled: warmup 300, decay to 0.3x at 4000, 0.06x until 6000 |
| batch / budget | 800 samples x 45k steps on 32 H100 (~48 h, ~120 epochs) | 4 GPUs (shard 4) x 128 windows = 512 windows x 16 actions per step, 6000 iterations = 3.07 M windows ~ 10 epochs of the ~300 k training windows; ~55 s/iter on 4 x A40 (RoboCasa365 measurement at this geometry) -> **~4 days per run** |
| validation | none | none by default (a seeded 5 % in-sample monitor exists: `[trainer].run_validation=true`) |
| eval | 50 trials x 24 tasks x 3 seeds, 5 denoising steps, deterministic | NOT ported yet (see below) |

baseinit vs reptileinit differ only in the init (`[checkpoint].load_path` = base DCP vs Reptile iteration dir +
`meta_action_init_*` into domain row 30 = `EMBODIMENT_TO_DOMAIN_ID["robocasa"]`) and the action-head LR multiplier
(5x fresh-head boost vs 1x), exactly like the LIBERO pairs. The 10D `[pos, rot6d, gripper]` contract is the layout of
every Reptile meta-training embodiment and of LIBERO, so the meta heads keep their column semantics.

## Files

| what | where |
|---|---|
| one-time data setup (download, HDF5 -> LeRobot v3, normalization stats) | `examples/setup_robocasa.sh` -> `cosmos_framework/scripts/convert_robocasa_cosmos_policy_to_lerobot_v30.py` |
| experiment | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robocasa_edge.py` (registered in `configs/base/config.py`) |
| recipes | `examples/toml/sft_config/action_policy_robocasa_edge_{baseinit,reptileinit}.toml` |
| launchers | `examples/launch_sft_action_policy_robocasa_edge_{baseinit,reptileinit}.sh` (ports 50900 / 50910) |
| Reptile meta-training (unchanged) | `examples/launch_reptile_meta_edge_v4.sh` + `action_reptile_meta_edge_v4.toml` |
| reader | `cosmos_framework/data/generator/action/datasets/robocasa_lerobot_dataset.py` (`DEFAULT_ROBOCASA_COSMOS_POLICY_TASKS`, `state_normalization`, `augment_*`) |
| stats | `cosmos_framework/data/generator/action/normalizer_stats/robocasa_cosmos_policy_frame_wise_relative_rot6d.json` (blocks `global` / `global_raw` / `state`) |
| tests | `cosmos_framework/configs/toml_config/robocasa_toml_test.py`, `.../datasets/robocasa_lerobot_dataset_augment_test.py` |

Setting this up on another machine: `docs/collaborator_setup.md` lists every external piece (checkpoints, data, envs, assets)
and where it comes from.

## 1. One-time data setup (login node, internet; CPU only)

```bash
cd ~/project/cosmos_robocasa/packages/cosmos3
export COSMOS_STORAGE=$HOME/project/cosmos_storage
bash examples/setup_robocasa.sh            # STEPS="download convert stats" (idempotent, resumable), JOBS=6
```

* `download`: `hf download nvidia/RoboCasa-Cosmos-Policy` (CC BY 4.0) -- only `success_only/kitchen_*/<Task>/<date>/*.hdf5`
  (24 files, ~144 GB of raw uint8 224x224 frames) + the dataset statistics JSONs -> `$COSMOS_STORAGE/data/robocasa_cosmos_policy`.
  The T5 / Reason1 embedding pickles (25 GB) and `all_episodes/` (world-model / value rollouts incl. failures) are skipped.
* `convert`: each HDF5 -> `$COSMOS_STORAGE/data/robocasa_cosmos_policy_lerobot_v3/<Task>/<date>/lerobot` (lerobot 0.4.x API,
  AV1 videos per camera, parquet) with the RoboCasa365 column contract (12D action with the base channels zero-filled and
  `control_mode=-1`, 16D state with an identity base pose, per-episode `task` text), plus `extras/stats_arrays.npz`
  (the reader's 10D actions / state tokens) and `extras/task_meta.json`. `JOBS` tasks run in parallel.
* `stats`: aggregates the arrays of the 24 shards into the bundled normalizer JSON (commit it together with the recipes).

`ROBOCASA_ROOT` for training = the converted root. A single task converts in a few minutes; the whole set in ~1-2 h with
`JOBS=6`.

## 2. Training (4 x 48 GB)

```bash
cd ~/project/cosmos_robocasa/packages/cosmos3
export COSMOS_STORAGE=$HOME/project/cosmos_storage
export BASE_CHECKPOINT_PATH=$COSMOS_STORAGE/checkpoints/Cosmos3-Edge
export WAN_VAE_PATH=$COSMOS_STORAGE/checkpoints/wan22_vae/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=$COSMOS_STORAGE/outputs
export ROBOCASA_ROOT=$COSMOS_STORAGE/data/robocasa_cosmos_policy_lerobot_v3   # = the launcher default

# (1) mid-trained init (baseline)
NPROC_PER_NODE=4 sr 4 48 --exclude haring examples/launch_sft_action_policy_robocasa_edge_baseinit.sh

# (2) Reptile meta-init: the existing edge_reptile_v4 checkpoint is reusable (meta-training never sees RoboCasa);
#     otherwise run examples/launch_reptile_meta_edge_v4.sh first
JOB=$COSMOS_STORAGE/outputs/cosmos3_action_meta/reptile_meta/edge_reptile_v4
export REPTILE_CKPT_PATH=$JOB/checkpoints/iter_000001000
export META_ACTION_INIT_PATH=$JOB/meta_action_init_iter_001000.pt
NPROC_PER_NODE=4 sr 4 48 --exclude haring examples/launch_sft_action_policy_robocasa_edge_reptileinit.sh
```

Runs land in `$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_robocasa/edge_robocasa_h50/edge_robocasa_h50_fullft_<init>/`
(DCP at 2000 / 4000 / 6000, ~24 GB each; `[checkpoint].save_iter`). Re-running the same launcher auto-resumes.
Smoke idiom: `EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 job.wandb_mode=disabled"` with a throw-away
`IMAGINAIRE_OUTPUT_ROOT` (a final DCP is always written).

Knobs that are deliberately experiment-level (override through `EXTRA_TAIL_OVERRIDES`, both loader nodes):

```bash
# all three cameras (wrist on top, left/right half size below) instead of the LIBERO 2-view canvas
EXTRA_TAIL_OVERRIDES="dataloader_train.dataloader.datasets.robocasa.dataset.camera_set=wrist_lr dataloader_val.dataloader.datasets.robocasa.dataset.camera_set=wrist_lr"
# Cosmos3's default prompt dropout (CFG at eval) instead of Cosmos Policy's none
EXTRA_TAIL_OVERRIDES="dataloader_train.dataloader.datasets.robocasa.dataset.cfg_dropout_rate=0.1"
```

Budget: `trainer.max_iter` and `scheduler.cycle_lengths[0]` (keep it at ~2/3 of `max_iter`) scale the run; the 2000-step
DCPs give the 2000 / 4000-iteration points of the same schedule for free only up to 4000 (the 0.06x tail starts there).

## 3. Closed-loop evaluation (ORIGINAL RoboCasa, v0.2 simulator, Cosmos Policy protocol)

One-time simulator setup (login node; pip installs code from an external repo, so run it yourself):

```bash
bash examples/setup_robocasa_v02_eval.sh      # STEPS="env assets": conda env robocasa-v02-eval + ~/project/robocasa_cosmos_policy + 5 GB assets
```

`env` = python 3.10 + robosuite 1.5.1 + mujoco 3.2.6 (Cosmos Policy's pins) + the Cosmos Policy fork of robocasa v0.2
(`moojink/robocasa-cosmos-policy` = v0.2 with its numpy assert removed; numpy 1.23.3 / numba 0.56.4 from its setup.py)
+ requests / imageio-ffmpeg / scipy; `assets` = the v0.2 kitchen assets from the utexas Box links + `macros_private.py`.
The RoboCasa365 env (`robocasa-eval`) cannot be reused: different package, different assets.

Evaluation (1 GPU; ~24 tasks x 50 rollouts):

```bash
cd ~/project/cosmos_robocasa/packages/cosmos3
RUN=edge_robocasa_h50_fullft_baseinit            # or edge_robocasa_h50_fullft_reptileinit
RUN_ROOT=$COSMOS_STORAGE/outputs/cosmos3_action_robocasa/edge_robocasa_h50/$RUN ITER=6000 \
OUT_DIR=$COSMOS_STORAGE/outputs/eval_robocasa/${RUN}_iter6000_seed195 \
    sr 1 48 --exclude haring examples/eval_robocasa_closed_loop.sh
```

The script starts `action_policy_server_robocasa` (cosmos3-pt env, EMA weights, chunk 16 from the run's config.yaml,
`--action-stats-path` = the bundled stats + `--action-normalization quantile_rot`, so the 10D actions come back
denormalized; the JSON prompt now carries the training-time view sentence of `camera_set`), waits for `/info`, then runs
`cosmos_framework/simulation/robocasa_v02/closed_loop_eval.py` per task in the simulator env. Protocol = Cosmos Policy's
`run_robocasa_eval.py` (`protocol.py` holds the pure-python pieces): PandaMobile + the OSC_POSE controller the demos used
(`osc_pose_controller_config.json`), object split "B", the 5 official test scenes `((1,1),(2,2),(4,4),(6,9),(7,10))` with
episodes 0-9 / 10-19 / ... per scene, one fresh env per episode seeded `SEED * episode * 256` (`DETERMINISTIC=1`), 10
zero-action settling steps, the registry horizon per task (300-1000), first `_check_success()` ends the episode; images =
the 224x224 cameras flipped upright -> `[agentview_left | eye_in_hand]` 224x448; state token = `[eef_pos, rot6d(eef_quat),
qpos0 - qpos1]` min/max-normalized with the stats JSON's `state` block (the client normalizes, the server prepends it
raw); 16 actions executed per query (Cosmos Policy predicts 32 / executes 16), decoded to the 12D env action
`[pos, rotvec, gripper, 0, 0, 0, 0, -1]`. Per task: `<OUT_DIR>/<task>/results.json` (progressive), `rolloutNN.mp4`,
`rolloutNN_init.png`, `eval.log`; overall `<OUT_DIR>/summary.json` (`mean_task_success_rate` = the paper's metric).
Tasks with a complete results.json are skipped (resume). Cosmos Policy reports the mean over `SEED=195 196 197`;
`NUM_TRIALS=10 TASKS="OpenDrawer CloseDrawer"` is the quick first pass, `NO_VIDEO=1` skips the mp4s.

Noise floor: 50 rollouts/task -> per-task SE ~0.07, 24-task mean SE ~0.015 per seed.

Not exercised on a GPU yet: the server/client pair was validated statically (16 tests: rot6d / state-token / decode round
trips against the training reader and converter, Cosmos Policy schedule, script wiring); the first real run should start
with one task and 2-3 rollouts (`NUM_TRIALS=3 TASKS=OpenDrawer`) and a look at `server.log` + `rollout00_init.png`.
