# RoboCasa365 post-training of Cosmos3-Edge (cosmos_hs11)

Mid-trained Cosmos3-Edge (or its Reptile meta-init, `docs/action_reptile_meta.md`) post-trained on the RoboCasa365
`target/atomic` split and evaluated closed-loop in the RoboCasa simulator. The setting is the **"Target only"** row of
RoboCasa365 Table 2 (ICLR 2026): no exposure to the RoboCasa365 pretrain split, only the target demos, at the paper's
three data fractions. Published reference (GR00T N1.5, target only, 30 trials/task): 38.7 / 50.6 / 60.6 % at
50 / 150 / 500 demos per task.

| | |
|---|---|
| tasks | the 18 `target/atomic` tasks (PandaOmron mobile manipulator, 10 target kitchens, `target` object split) |
| demos | ~500 human demos/task (the exports hold 502-513); seeded 1 % val hold-out = 5/task; train = 50 (10 %), 150 (30 %) or every remaining demo (~500, 100 %) per task |
| action | 15D `base_encoding="raw_armfirst"`: `[eef_pos(3), eef_rot6d(6), gripper(1), base_motion(4), control_mode(1)]`, no normalization. Dims 0..9 = the 10D `[pos, rot6d, gripper]` layout of every Reptile meta-training embodiment (fractal / bridge / RoboMIND-UR / RoboMIND-Franka; YAM = left + right) and of LIBERO, so the meta heads keep their column semantics; base + mode (mostly 0 / -1) sit on dims 10..14, which YAM's right arm trained |
| observation | `agentview_left` \| `eye_in_hand` at native 256x256 each (256x512, the LIBERO concat-view geometry) + EEF pose token (`use_state`) |
| window | `chunk_length=16` (17 frames at 20 fps) = the LIBERO / Reptile contract; eval re-plans every 16 steps |
| batch | 4 GPUs (shard 4) x 128 windows (`max_samples_per_batch`, binding: token packing is off, `max_num_tokens_after_packing` is only read by the cost model) = 512 windows x 16 actions per step. The LIBERO Edge TOML runs 128 x 2 ranks = 256. Measured on the baseinit 10 % run: ~1.9 k tokens/window, 238 k tokens/rank/iter, peak 29 GB/GPU (48 GB A40), 55 s/iter of which 34.5 s is VAE encoding of the 128 clips; a full run takes ~46 h |
| schedule | 3000 iterations for every fraction (as the paper: one step budget), warmup 200, linear decay, val every 500. Coverage: the 18 exports hold 9126 episodes / 2.23 M frames (245 frames/episode on average), i.e. ~206 k / 617 k / 2.04 M 17-frame windows for 10 / 30 / 100 %, so 3000 x 512 windows = 7.5 / 2.5 / 0.75 epochs |
| eval | official Atomic-Seen protocol, 30 rollouts/task by default (`NUM_TRIALS`; fixed env seed 0 -> identical scenes for every checkpoint) |

Measured LIBERO Edge speed with the same batch geometry: ~29 s/iteration on 4 x 48 GB (peak ~40 GB/GPU), so one run is
~24 h; the six runs are ~6 days of 4-GPU time. `trainer.max_iter` / `scheduler.cycle_lengths` are the knobs.

## Files

| what | where |
|---|---|
| one-time setup (simulator env, assets, datasets, v3.0 conversion) | `examples/setup_robocasa365.sh` -> `cosmos_framework/scripts/download_robocasa365.py`, `cosmos_framework/scripts/convert_robocasa_to_lerobot_v30.sh` |
| experiment | `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robocasa_edge.py` (registered in `configs/base/config.py`) |
| recipes | `examples/toml/sft_config/action_policy_robocasa_edge_{baseinit,reptileinit,reptileinit_freshheads}_{10,30,100}pct.toml` |
| launchers | `examples/launch_sft_action_policy_robocasa_edge_{baseinit,reptileinit,reptileinit_freshheads}.sh` (`DATA_FRACTION=10|30|100`; ports 50800-50802 / 50810-50812 / 50820-50822) |
| data fraction | `RoboCasaLeRobotDataset(episodes_per_task=, episode_subset_seed=)` -> `_filter_valid_episodes` (per task shard, seeded, nested); TOML `[dataloader_train].episodes_per_task` -> `dataloader_train.dataloader.datasets.robocasa.dataset.episodes_per_task` |
| eval | `examples/eval_robocasa_closed_loop.sh` -> `cosmos_framework/scripts/action_policy_server_robocasa.py` (cosmos3-pt env) + `cosmos_framework/simulation/robocasa/closed_loop_eval.py --base-encoding raw_armfirst` (robocasa-eval env; `eval_utils.decode_15d_armfirst_to_env12`) + `cosmos_framework/simulation/robocasa/summarize_results.py` |
| tests | `cosmos_framework/configs/toml_config/robocasa_toml_test.py` |

Baseinit vs reptileinit differ only in the init (`[checkpoint].load_path`, `meta_action_init_*` into domain row 30 =
`EMBODIMENT_TO_DOMAIN_ID["robocasa"]`) and the action-head LR multiplier (5x fresh-head boost vs 1x), exactly like the
LIBERO pairs. The three fractions differ only in `[job].name` and `[dataloader_train].episodes_per_task`.

Why arm-first: the Nano recipe's `"raw"` contract puts the base command first, which shifts the arm block to dims 5..14 and
scrambles the meta heads' column semantics (base -> `pos` columns, gripper -> a YAM-right-arm `rot6d` column). All five
meta-training readers emit `[pos_delta(3), rot6d_delta(6), gripper(1)]` (frame-wise relative, quantile-normalized; YAM twice),
and LIBERO emits the same order (native OSC deltas, `quantile_rot`), so `"raw_armfirst"` gives the RoboCasa arm block the same
alignment the LIBERO experiments had. The 5 extra channels have no meta-training counterpart; they go to dims 10..14 because
those columns were at least trained (YAM right arm) and their content is near-constant (base = 0 outside NavigateKitchen,
mode = -1 in arm mode). The base-first `"raw"` contract is untouched for the Nano recipe.

Third variant, `reptileinit_freshheads` (ablation): the Reptile DCP trunk with FRESH action heads and the 5x head LR of the
baseinit recipe (no `meta_action_init_*`). Together the three recipes separate the trunk's and the heads' share of the
Reptile-init effect: baseinit (fresh trunk + fresh heads) vs reptileinit_freshheads (Reptile trunk + fresh heads) vs
reptileinit (Reptile trunk + meta heads). Run it on the 10 % fraction first if compute is tight.

## 1. One-time setup (login node, internet)

```bash
cd ~/project/cosmos_hs11/packages/cosmos3
export COSMOS_STORAGE=$HOME/project/cosmos_storage
bash examples/setup_robocasa365.sh                       # STEPS="env assets data convert" (idempotent, resumable)
```

Steps `assets` and `data` fetch from the official UT Austin Box links. Some clusters reset every TCP connection to
`utexas.box.com` (`[Errno 104] Connection reset by peer` on every archive, 3/3 attempts); `DOWNLOAD_SOURCE=auto`
(default) then falls back to the Hugging Face mirrors `twilighted/Robocasa365-Assets` (6 zips, byte-identical to Box)
and `huiwon/robocasa365` (same `<split>/<kind>/<Task>/<date>/lerobot.tar` tree; tar metadata differs, extracted files
identical -- both checked 2026-11-02). `DOWNLOAD_SOURCE=hf bash examples/setup_robocasa365.sh` skips Box altogether
(`cosmos_framework/scripts/download_robocasa365.py --source`).

* `env`: conda env `robocasa-eval` (python 3.11) with `~/project/robosuite` (ARISE master, 1.5.2) and `~/project/robocasa`
  (RoboCasa365 1.0.1) as editable installs + `requests imageio-ffmpeg`; writes `robocasa/macros_private.py` with
  `DATASET_BASE_PATH=$COSMOS_STORAGE/data/robocasa365`. robocasa pins numpy 2.2.5 / mujoco 3.3.1 / lerobot 0.3.3, hence
  the separate env (cosmos-framework needs lerobot 0.4.x).
* `assets`: kitchen assets (textures, generative textures, lightwheel fixtures, objaverse + aigen + lightwheel objects)
  into `~/project/robocasa/robocasa/models/assets`. The target demos use the default `("objaverse", "lightwheel")`
  registries.
* `data`: the 18 `target/atomic` tars (LeRobot v2.1 + `extras/dataset_meta.json`, one dated export per task) ->
  `$COSMOS_STORAGE/data/robocasa365/v1.0/target/atomic/<Task>/<date>/lerobot`.
* `convert`: v2.1 -> v3.0 copies -> `$COSMOS_STORAGE/data/robocasa365/v30/target/atomic/<Task>/<date>/lerobot`
  (`ROBOCASA_ROOT` for training; the eval keeps reading the v2.1 dirs).

## 2. Training (4 x 48 GB)

```bash
cd ~/project/cosmos_hs11/packages/cosmos3
export COSMOS_STORAGE=$HOME/project/cosmos_storage
export BASE_CHECKPOINT_PATH=$COSMOS_STORAGE/checkpoints/Cosmos3-Edge
export WAN_VAE_PATH=$COSMOS_STORAGE/checkpoints/wan22_vae/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=$COSMOS_STORAGE/outputs
export ROBOCASA_ROOT=$COSMOS_STORAGE/data/robocasa365/v30/target/atomic   # = the launcher default

# (1) mid-trained init, 10 / 30 / 100 % of the target demos
DATA_FRACTION=10  NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_robocasa_edge_baseinit.sh
DATA_FRACTION=30  NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_robocasa_edge_baseinit.sh
DATA_FRACTION=100 NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_robocasa_edge_baseinit.sh

# (2) Reptile meta-init, same three fractions
JOB=$COSMOS_STORAGE/outputs/cosmos3_action_meta/reptile_meta/edge_reptile_v4
export REPTILE_CKPT_PATH=$JOB/checkpoints/iter_000001000
export META_ACTION_INIT_PATH=$JOB/meta_action_init_iter_001000.pt
DATA_FRACTION=10  NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_robocasa_edge_reptileinit.sh
DATA_FRACTION=30  NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_robocasa_edge_reptileinit.sh
DATA_FRACTION=100 NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_robocasa_edge_reptileinit.sh

# (3) ablation: Reptile trunk + fresh heads (5x head LR), same three fractions (REPTILE_CKPT_PATH only)
DATA_FRACTION=10  NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_robocasa_edge_reptileinit_freshheads.sh
```

Runs land in `$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_robocasa/edge_robocasa365_atomic/edge_robocasa365_atomic_<frac>pct_fullft_<init>/`
(DCP every 1000 iterations, ~24 GB each; `[checkpoint].save_iter`). Re-running the same launcher auto-resumes.
`EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 trainer.run_validation_on_start=false"` is the smoke idiom.

### Image augmentation (`[dataloader_train].use_image_augmentation`, 2026-11-06)

`use_image_augmentation = true` in a TOML turns on the DROID-recipe train-time augmentation in `RoboCasaLeRobotDataset`
(ported from `DROIDLeRobotDataset`, the same numbers as GR00T N1.5's RoboCasa365 data config and NVIDIA's own DROID
policy post-training): random crop to 95 % of each view + resize back + ColorJitter(brightness 0.3, contrast 0.4,
saturation 0.5, hue 0.08). One parameter draw per window is applied to all 17 frames of both views (the views are
stacked along T before the transform; the crop never runs across the tile seam). The key is remapped to the
`robocasa` TRAIN dataset node only; the val loader never augments (the reader also ignores the flag for
`split != "train"`). Off by default, so the v1 runs (`*_10pct.toml` etc.) are unchanged; `*_baseinit_10pct_v2.toml`
has it on. Unit tests: `robocasa_lerobot_dataset_augment_test.py` + the routing test in `robocasa_toml_test.py`.

### Flow-matching loss weights (`[model.rectified_flow_training_config]`, 2026-11-06)

`action_loss_weight`, `loss_scale` and `normalize_loss_by_active` are now TOML keys (section
`[model.rectified_flow_training_config]`, routed to `model.config.rectified_flow_training_config.*`; omitted keys keep the
experiment's values 10 / 10 / false). The `*_baseinit_*pct_v2.toml` recipes set `action_loss_weight = 100.0`: the Cosmos 3
report calibrated the x10 for NORMALISED actions, while these recipes regress un-normalised 15D actions whose per-element
MSE is ~10x smaller (5 constant dims, near-identity rot6d), so at 10 the action term is ~9 % of the objective against
the vision term and at 100 roughly 50/50. Untested as an ablation so far; `--dryrun` shows the composed value.

## 3. Closed-loop evaluation (1 GPU; 30 rollouts/task by default)

```bash
cd ~/project/cosmos_hs11/packages/cosmos3
export COSMOS_STORAGE=$HOME/project/cosmos_storage
RUN=edge_robocasa365_atomic_10pct_fullft_baseinit
RUN_ROOT=$COSMOS_STORAGE/outputs/cosmos3_action_robocasa/edge_robocasa365_atomic/$RUN ITER=3000 \
OUT_DIR=$COSMOS_STORAGE/outputs/eval_robocasa/${RUN}_iter3000 \
    sr 1 48 examples/eval_robocasa_closed_loop.sh
```

The script starts `action_policy_server_robocasa` (DCP + the run's `config.yaml`, EMA weights, chunk 16 read from the
config), waits for `/info`, then runs the client per task with the fixed contract flags (`--action-horizon 16
--camera-set left_wrist --use-state --use-base-action --base-encoding raw_armfirst`), `--seed 0`, the official per-task horizon
(robocasa v1.0.1 registry) and first-success latching. Per task: `<OUT_DIR>/<task>/results.json`, `rolloutNN.mp4`,
`rolloutNN_init.png`, `eval.log`; overall: `<OUT_DIR>/summary.json` (`mean_task_success_rate` = the RoboCasa365
task-averaged metric, plus pooled counts). Tasks with an existing `results.json` are skipped (resume). `NUM_TRIALS=10`
is a quicker first pass, `TASKS="CloseFridge OpenDrawer"` restricts the set, `SAVE_GEN_VIDEO=1` also stores
the model's generated video per rollout.

With 30 rollouts per task (540 episodes) the per-task standard error is ~0.09 and the 18-task mean's ~0.02; with
`NUM_TRIALS=10` they are ~0.15 and ~0.04, so differences below ~0.1 mean success are not resolvable at 10 trials.

Notes:
- `action_policy_server_robocasa` builds `OmniInference` with `guardrails = False` (same as the LIBERO server). The
  NVIDIA original left the Cosmos safety guardrails on, which made the server download the gated
  `nvidia/Cosmos-Guardrail1` HF repo at startup and die with `Access denied. This repository requires approval`
  (`<OUT_DIR>/server.log`); action-policy evaluation never needs them.
- The eval script exports `NCCL_IB_DISABLE=1` like the training launcher: on nodes `bob` / `greco` (broken IB/RoCE
  stack) the server's first collective, the 1-rank `broadcast_object_list` inside `OmniInference.create`, otherwise
  dies with `Fatal Python error: Segmentation fault` in `distributed_c10d.broadcast` (seen on bob 2026-11-01).
- Training launchers hard-code one `MASTER_PORT` per recipe (copies such as `*_v2.sh` / `*_v3.sh` keep the same
  port). `_sft_launcher_common.sh` now probes the port on the node and, if it is already taken, uses the next free one
  (printed as `MASTER_PORT: <port>`; `MASTER_PORT_AUTO=0` keeps the exact port). Multi-node launches are not touched.
- In an interactive zsh, `RUN=name   # comment` is NOT an assignment plus a comment (zsh runs `#` as a command and
  drops the assignment, so `$RUN` stays empty and `OUT_DIR` becomes `.../eval_robocasa/_iter3000`). Keep the
  assignment on its own line without a trailing comment, or `setopt interactive_comments`.

## 3b. Train/eval consistency check by demo replay (CPU, ~1 min per demo)

`cosmos_framework/simulation/robocasa/replay_consistency_check.py` (robocasa-eval env, `MUJOCO_GL=osmesa` works on a
CPU node) restores recorded demos in the exact eval environment (`closed_loop_eval.make_env`, Atomic-Seen) from the
LeRobot v2.1 `extras/episode_XXXXXX/{ep_meta.json, model.xml.gz, states.npz}` and compares the eval side with the
dataset: first observation vs video frame 0 (pixel diff), `build_state_token` vs `observation.state`, the prompt
`get_ep_meta()["lang"]` vs the annotation, and a replay of the recorded actions through `env.step`, both reordered to
the env order and through the reader's 15D arm-first encoding + `decode_15d_armfirst_to_env12` (the closed-loop path),
with eef / gripper drift against the recorded states and `_check_success`. Frames come from
`extract_demo_frames.py` (cosmos3-pt env, PyAV):

```bash
python cosmos_framework/simulation/robocasa/extract_demo_frames.py $ROBOCASA_V21_ROOT <out> '{"TurnOnMicrowave": [0, 100]}'
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=. $ROBOCASA_PY \
  cosmos_framework/simulation/robocasa/replay_consistency_check.py $ROBOCASA_V21_ROOT <out> '{"TurnOnMicrowave": [0, 100]}'
```

2026-11-08, 6 demos of TurnOnMicrowave / PickPlaceCounterToCabinet / CloseBlenderLid: pixel diff 1.6-4.0 / 255 (19 with
`translucent_robot=True`, i.e. the videos show the opaque robot the eval renders), state-token error 0, prompts
identical, replay eef drift ~1 mm and gripper 0 with success at the recorded step in 5/6 (CloseBlenderLid ep0 drifts to
2.5 cm on a knife-edge placement). So rendering, proprioception, language, action semantics, controller and control
rate agree between training data and the closed loop; the low success rate is the policy. Pitfall when writing such a
replay: the dataset stores actions base-first `[base4, mode, pos3, rot3, grip]` while `env.step` expects
`[pos3, rot3, grip, base4, mode]`; feeding the stored order drifts 0.5-1 m.

## 4. Open-loop accuracy (no simulator; 4 GPUs, ~6 min)

The closed-loop success rate of these recipes is near the floor (baseinit 10 %: 7.6 % mean over 18 tasks x 30 rollouts
on 2026-11-02; the collaborator's 10 % freshheads / 30 % baseinit / 30 % freshheads / 100 % baseinit: 7 / 3.3 / 1.7 /
2.4 %), while the validation flow-matching loss (0.014-0.02) looks fine. The FM loss is a velocity-regression error at
random noise levels and says nothing about the SAMPLED chunk, so `cosmos_framework/simulation/robocasa/openloop_eval.py`
measures that directly: for 12 windows per task it sends the first frame + language + EEF state token of a demo window
to the normal policy server and compares the sampled 16-step chunk with the demonstrator's actions in the env's 12D
space (`decode_15d_armfirst_to_env12`), against two trivial baselines (zero action, per-task mean action).

```bash
RUN_ROOT=$COSMOS_STORAGE/outputs/cosmos3_action_robocasa/edge_robocasa365_atomic/$RUN ITER=3000 \
  sr 4 48 --qos=<qos> examples/eval_robocasa_openloop.sh                      # SPLIT=val (held-out demos, default)
SPLIT=train EPISODES_PER_TASK=50 RUN_ROOT=... sr 4 48 examples/eval_robocasa_openloop.sh   # the run's own training demos
```

One `action_policy_server_robocasa` per allocated GPU (ports 8901+); results in
`$COSMOS_STORAGE/outputs/eval_robocasa_openloop/<run>_iter<ITER>_<split>/openloop_summary.json` (+ `openloop.log`,
`openloop_predictions.npz`). `~/.local/bin` (uv) must be on PATH, the server's HF download step shells out to it.
Reading: small errors + high correlation = the policy fits the demos and closed-loop failure is compounding / precision;
errors at the baseline level = the action head never fit the demos (recipe problem). val vs train separates
underfitting (both bad) from overfitting (train good, val bad).

Result for baseinit 10 % iter 3000, val split (216 windows, 2026-11-06): pos MAE 0.234 vs zero-action 0.242 vs
per-task-mean 0.234 (R^2 vs task mean 0.07, corr 0.60); rot MAE 0.078, WORSE than zero (0.069); chunk-mean pos MAE
0.19 vs task-mean 0.21; gripper sign accuracy 0.96 overall but 0.73 on windows with a gripper transition; |pred| ~ |gt|
(not sluggish); error flat across the 16 steps; the sampler is deterministic (two queries of one input agree exactly).
Variance structure of the 15D training actions (what the FM loss sees): gripper 60 %, eef_pos 29 %, rot6d 6 %,
base + mode 5 % (five constant dims), so the action term (`action_loss_weight` 10 x ~0.009) is ~9 % of the training
objective against the vision term (10 x ~0.094); on the LIBERO recipe (quantile_rot-normalised 10D actions) the two
terms are of the same order. `action_loss_weight` / `normalize_loss_by_active` are not TOML keys but compose through
the launcher's tail overrides, e.g.
`EXTRA_TAIL_OVERRIDES="model.config.rectified_flow_training_config.action_loss_weight=100"`.
Same checkpoint on its OWN training demos (`SPLIT=train EPISODES_PER_TASK=50`, 216 windows): pos MAE 0.172 vs
zero-action 0.268 vs task-mean 0.261 (corr 0.82), rot MAE 0.063 < zero 0.071, gripper transition accuracy 0.81. So the
head does fit the demos it was trained on and loses almost all of it on new demos of the same tasks and kitchens
(val corr 0.60, R^2 0.07): the 10 % run (7.5 epochs over 50 demos/task) memorises trajectories instead of learning a
visually grounded policy; it is a generalisation failure, not "not converged".
