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
