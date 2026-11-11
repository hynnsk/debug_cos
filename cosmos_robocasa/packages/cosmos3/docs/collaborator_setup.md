# Setting up cosmos_robocasa on another machine (collaborator checklist)

Everything the RoboCasa recipes need that is NOT in this repository, and where each piece comes from. All of it is
publicly downloadable except the Reptile meta-checkpoint (item 4). Paths below assume
`COSMOS_STORAGE=<some big disk>/cosmos_storage`; every launcher reads its inputs from env vars, so any layout works.

| # | what | source | where it goes | needed for |
|---|---|---|---|---|
| 1 | training env (the existing `cosmos3-pt`-like env) | nothing new to install: `h5py` and `lerobot` (mli0603 fork, 0.4.4) are already `pyproject.toml` dependencies and the PyAV wheel bundles its own ffmpeg with `libsvtav1`. Check with `python -c "import h5py, lerobot, av; assert 'libsvtav1' in av.codecs_available"`; only if that fails, `pip install h5py` / reinstall the env per `docs/setup.md` | conda env | training, data conversion, policy server |
| 2 | Cosmos3-Edge mid-trained DCP | `python -m cosmos_framework.scripts.convert_model_to_dcp` with `BASE_CHECKPOINT_NAME=Cosmos3-Edge` (downloads the public `nvidia/Cosmos3-Edge` from the Hub, `docs/training.md` Step 2); 6.3 GB | `$COSMOS_STORAGE/checkpoints/Cosmos3-Edge` = `BASE_CHECKPOINT_PATH` | baseinit training |
| 3 | Wan2.2 VAE | `hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir $COSMOS_STORAGE/checkpoints/wan22_vae`; 2.7 GB | `WAN_VAE_PATH` | all training / serving |
| 4 | Reptile meta-checkpoint (`edge_reptile_v4`, iter 1000: DCP `checkpoints/iter_000001000/model/` 24 GB + `meta_action_init_iter_001000.pt`) | **not downloadable**: a training output. Either (a) get it from the author (24 GB DCP dir + the .pt), (b) re-run `examples/launch_reptile_meta_edge_v4.sh` with the same TOML (needs the robot_fewshot source data, ~1-2 days on 4 GPUs; not bit-identical), or (c) use your own Reptile run (`edge_reptile_v4.x`) and say which | `REPTILE_CKPT_PATH`, `META_ACTION_INIT_PATH` | reptileinit training |
| 5 | RoboCasa training data (Cosmos Policy export) | `bash examples/setup_robocasa.sh` -> `hf download nvidia/RoboCasa-Cosmos-Policy` (public, CC BY 4.0; only `success_only/kitchen_*`, 144 GB) + HDF5 -> LeRobot v3 conversion (3.7 GB, ~1 h with JOBS=6) | `$COSMOS_STORAGE/data/robocasa_cosmos_policy` (raw) and `$COSMOS_STORAGE/data/robocasa_cosmos_policy_lerobot_v3` = `ROBOCASA_ROOT` | training |
| 6 | normalization stats JSON | **in the repo** (`cosmos_framework/data/generator/action/normalizer_stats/robocasa_cosmos_policy_frame_wise_relative_rot6d.json`); `STEPS="download convert"` is enough, the `stats` step only regenerates the identical file | - | training + eval |
| 7 | simulator env for evaluation (`robocasa-v02-eval`: python 3.10, robosuite 1.5.1, mujoco 3.2.6, Cosmos Policy's robocasa v0.2 fork) | `bash examples/setup_robocasa_v02_eval.sh` (`STEPS=env`: PyPI + `git clone moojink/robocasa-cosmos-policy`) | conda env + `~/project/robocasa_cosmos_policy` | eval |
| 8 | v0.2 kitchen assets (4 zips, 3.7 GB) | same script, `STEPS=assets`: utexas Box via the fork's downloader, or `ASSET_SOURCE=hf` = the byte-identical Hugging Face mirror `Tej3/robocasa-v02-kitchen-assets` (SHA256-verified) for clusters that cannot reach Box | `~/project/robocasa_cosmos_policy/robocasa/models/assets` | eval |

Not needed: LIBERO data, the robot_fewshot source data (only for item 4b), any gated Hugging Face repo (the servers run
with the Cosmos guardrails off), `LIBERO_ROOT`.

## Commands, in order

The setup scripts default to `$HOME/anaconda3/...` interpreters; on another machine pass `TRAIN_PY`, `CONDA_BASE` and
`ROBOCASA_PY` as below. Steps 1-3 run once on a node with internet; the training launchers use the `torchrun` of the
active env.

```bash
# 0. shell (every session)
conda activate <training env>
cd <clone>/cosmos_robocasa/packages/cosmos3            # launchers derive WORKDIR from their own location; PYTHONPATH=. is set by them
export COSMOS_STORAGE=<big disk>/cosmos_storage
export BASE_CHECKPOINT_PATH=$COSMOS_STORAGE/checkpoints/Cosmos3-Edge
export WAN_VAE_PATH=$COSMOS_STORAGE/checkpoints/wan22_vae/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=$COSMOS_STORAGE/outputs

# 1. item 1: env check (no install expected)
python -c "import h5py, lerobot, av; assert 'libsvtav1' in av.codecs_available; print('ok')"

# 2. items 5-6: download 144 GB + convert to 3.7 GB LeRobot v3 (~1.5-2 h); the stats JSON is already in the repo
STEPS="download convert" TRAIN_PY=$(which python) JOBS=6 bash examples/setup_robocasa.sh

# 3. items 7-8: simulator env + v0.2 kitchen assets (Hugging Face mirror when utexas.box.com is blocked)
CONDA_BASE=$(conda info --base) ASSET_SOURCE=hf bash examples/setup_robocasa_v02_eval.sh

# 4. GPU geometry: only when NOT on 4 GPUs (2 GPUs -> shard 2 x 256 windows, global batch stays 512)
sed -i -e 's/^\(data_parallel_shard_degree *= *\)4/\12/' -e 's/^\(max_samples_per_batch *= *\)128/\1256/' \
    examples/toml/sft_config/action_policy_robocasa_edge_{baseinit,reptileinit}.toml

# 5. training (NPROC_PER_NODE = shard degree in the TOML)
NPROC_PER_NODE=4 bash examples/launch_sft_action_policy_robocasa_edge_baseinit.sh
REPTILE_CKPT_PATH=<reptile job>/checkpoints/iter_000001000 \
META_ACTION_INIT_PATH=<reptile job>/meta_action_init_iter_001000.pt \
NPROC_PER_NODE=4 bash examples/launch_sft_action_policy_robocasa_edge_reptileinit.sh

# 6. evaluation per checkpoint (1 GPU with EGL; resumes per task)
ROBOCASA_PY=$(conda info --base)/envs/robocasa-v02-eval/bin/python \
RUN_ROOT=$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_robocasa/edge_robocasa_h50/edge_robocasa_h50_fullft_baseinit ITER=6000 \
OUT_DIR=$COSMOS_STORAGE/outputs/eval_robocasa/baseinit_iter6000_seed195 bash examples/eval_robocasa_closed_loop.sh
```

## Different GPU count / memory

The TOMLs are written for 4 ranks x 128 windows (global batch 512). Keep the global batch when changing the geometry:
2 x 80 GB -> `[model.parallelism].data_parallel_shard_degree = 2` and `[dataloader_train].max_samples_per_batch = 256`
(the RoboCasa365 recipes ran 256/rank on H200), launched with `CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2`. The launcher
warns when `NPROC_PER_NODE` disagrees with the TOML's shard degree (the framework would silently run HSDP and double the
batch). `trainer.max_iter` / `scheduler.cycle_lengths[0]` (keep ~2/3 of max_iter) scale the budget.

## Sanity checks

* `PYTHONPATH=. python -m pytest --noconftest -c /dev/null -p no:cacheprovider --rootdir=. cosmos_framework/configs/toml_config/robocasa_toml_test.py cosmos_framework/simulation/robocasa_v02/protocol_test.py` (CPU, ~30 s).
* `PYTHONPATH=. python -m cosmos_framework.scripts.train --sft-toml examples/toml/sft_config/action_policy_robocasa_edge_baseinit.toml --dryrun -- job.wandb_mode=disabled` composes the config without a GPU (env vars above must be set).
* First real run: `EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 job.wandb_mode=disabled"` with a throw-away `IMAGINAIRE_OUTPUT_ROOT`; first eval: `NUM_TRIALS=3 TASKS=OpenDrawer`.

See `docs/action_policy_robocasa_posttrain.md` for the recipe itself.
