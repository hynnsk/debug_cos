# Cross-embodiment few-shot meta-training of the Cosmos3-Edge action heads (cosmos_hs07)

This repo adds a **meta-training stage between Cosmos3 mid-training and the LIBERO post-training**:

```
Cosmos3-Edge (mid-trained)                       baseline: fresh action heads ──┐
      │                                                                          ▼
      ├─► cross-embodiment FOMAML on the action I/O projectors ─► meta_action_init.pt ─► LIBERO-10 few-shot
      │   (RT-1, Bridge V2, RoboMIND-UR, RoboMIND-Franka, MolmoAct2-YAM)              post-training (3 demos/task)
      └─► backbone frozen                                                                 + closed-loop eval
```

Hypothesis: the Cosmos3 representation is already good; what a new embodiment needs is a **rapidly
adaptable action interface**. So only `action2llm`, `llm2action` and `action_modality_embed` are
meta-learned — one *shared* initialization `theta_meta` (not five independent domain rows), trained with
first-order MAML so that a few demonstrations of an unseen embodiment adapt it quickly. Downstream, the
LIBERO recipe is unchanged except that `theta_meta` replaces the fresh init of the LIBERO domain row.

| Piece | Path |
| --- | --- |
| YAM reader (14-D joints → FK → 20-D dual-arm rot6d) | `cosmos_framework/data/generator/action/datasets/molmoact2_yam_dataset.py` |
| YAM multi-repo wrapper (offset-aware blocks / episodes) | `cosmos_framework/data/generator/action/datasets/yam_repo_collection_dataset.py` |
| YAM forward kinematics (URDF, NumPy) + assets | `cosmos_framework/data/generator/action/yam_fk.py`, `robot_assets/yam.{urdf,xml}` |
| YAM normalization stats | `cosmos_framework/data/generator/action/normalizer_stats/molmoact2_yam_stats.json` (`tools/compute_action_stats_from_dataset.py`) |
| Memory-light rows + episode tables | `cosmos_framework/data/generator/action/meta/lazy_rows.py` |
| Meta-ready RT-1 / Bridge / UR / Franka readers + registry | `cosmos_framework/data/generator/action/meta/embodiments.py` |
| Episodic sampler + packed-batch loader | `cosmos_framework/data/generator/action/meta/episodic_sampler.py` |
| `theta_meta` adapter (fast weights, FOMAML bookkeeping, `.pt` I/O) | `cosmos_framework/data/generator/action/meta/meta_action_adapter.py` |
| Meta trainer | `cosmos_framework/scripts/train_action_meta.py` |
| Experiment / TOML / launcher | `configs/base/experiment/action/meta/action_fewshot_meta_edge.py`, `examples/toml/sft_config/action_fewshot_meta_edge.toml`, `examples/launch_meta_action_fewshot_edge.sh` |
| Downstream injection | `cosmos_framework/checkpoint/meta_action_init.py` (+ `checkpoint.meta_action_init_*` in `utils/config.py`, `toml_config/sft_config.py`, hook in `checkpoint/dcp.py`) |
| LIBERO "ours" recipe | `examples/toml/sft_config/action_policy_libero_10_edge_metainit.toml`, `examples/launch_sft_action_policy_libero_10_edge_metainit.sh` |
| Closed-loop eval helper | `examples/eval_libero_closed_loop.sh` |

## 1. Data

All five embodiments live under one root (`ROBOT_FEWSHOT_ROOT`, here
`$COSMOS_STORAGE/data/robot_fewshot`):

| key | dir | reader | raw action | video | fps |
| --- | --- | --- | --- | --- | --- |
| `fractal` | `google_robot_rt1` | `FractalMetaDataset` (RT-1, 87K eps, 599 tasks) | 10-D EE delta (pos3 + rot6d + grip) | ego 256×320 | 3 |
| `bridge` | `bridge_v2` | `BridgeMetaDataset` (50K eps, 22K instructions) | 10-D | image_0 480×640 | 5 |
| `robomind_ur` | `robomind/ur_1rgb` | `RoboMINDURMetaDataset` (25K eps, 114 tasks; MuJoCo FK on joint targets) | 10-D | camera_top 480×640 | 30 |
| `robomind_franka` | `robomind/franka_1rgb` | `RoboMINDFrankaMetaDataset` (5.3K eps, **2 tasks**) | 10-D | camera_top 720×1280 | 30 |
| `molmoact2_yam` | `yam/repos/*` | `YAMRepoCollectionDataset` of `MolmoAct2YAMDataset` (100 repos, 4.6K eps, 18 instructions) | **20-D** dual-arm | top + wrists composite 540×640 | 30 |

The four existing readers are reused for the action conversion; the meta-ready subclasses only replace
`ActionBaseDataset._rows` (a `list[dict]` of every frame, tens of GB for RT-1/UR) with a column-store
`LazyRowTable`, and add `episode_table()` (per-demonstration window ranges + task ids). The upstream
`FractalLeRobotDataset.__init__` cannot even be instantiated in this release (it assigns to the read-only
`_rows` property) — `FractalMetaDataset` re-implements it.

**YAM.** MolmoAct2 stores 14-D absolute joints `[L joints0-5, L grip, R joints0-5, R grip]`. The
reader runs the i2rt YAM URDF chain (`yam_fk.py`, validated against MuJoCo on `yam.xml`) to get both
tool-centre poses, converts to frame-wise relative rot6d and emits
`[L pos3, L rot6d, L grip, R pos3, R rot6d, R grip]` (domain `molmoact2_yam` = 16). Gripper `1` = open,
which already matches the Cosmos convention (episodes start at ~1.0). The end-effector frame is the
`linear_4310` gripper `grasp_site` (+z along the fingers, i.e. `z = approach`); each arm's base
placement cancels out of the relative actions. Per-episode instructions come from
`meta/tasks_annotated.parquet`. Quantile stats were computed over all 100 repos:

```bash
PYTHONPATH=. python tools/compute_action_stats_from_dataset.py --embodiment molmoact2_yam \
  --data-root $ROBOT_FEWSHOT_ROOT -o cosmos_framework/data/generator/action/normalizer_stats/molmoact2_yam_stats.json
```

All embodiments go through the same `ActionTransformPipeline` as the LIBERO recipe (JSON prompts,
idle-frame metadata, cfg dropout 0.1, actions zero-padded to `max_action_dim=64` with
`raw_action_dim` masking, video snapped onto the `256` canvas tier). A 10-D and a 20-D embodiment
therefore share the same projection shapes, which is what makes one 64-D `theta_meta` meaningful.

## 2. Meta-episodes

`EpisodicEmbodimentSampler` draws, per meta-episode:

1. an embodiment **uniformly** (`p(e) = 1/5`, not proportional to dataset size);
2. `k_shot` support **demonstrations** and `q_query` query demonstrations of that embodiment
   (disjoint episodes; disjoint *tasks* whenever the embodiment has ≥ `min_tasks_for_disjoint`=4 tasks —
   RoboMIND-Franka with its 2 tasks falls back to episode-disjoint sampling);
3. `windows_per_demo` random 17-frame windows inside every chosen demonstration.

So `K` is a demonstration count — the same unit as the downstream "3 demos per task". Windows are
loaded in DataLoader workers and packed with `pack_samples_into_batch` into exactly the
`PackingDataLoader` batch layout `OmniMoTModel.training_step` consumes. Each (rank, worker) uses its
own seed, so the four ranks of a node work on four different meta-episodes.

## 3. Meta-training (`scripts/train_action_meta.py`)

```
for meta_iter:
    episode = next(loader)                                # one embodiment, support + query batches
    route every sample to the scratch DomainAwareLinear row (default 31)
    tokenize / VAE-encode support and query ONCE          (_get_training_inputs)
    fast <- theta_meta                                    (adapter.start_episode)
    repeat inner_steps: loss(support) -> grad(fast) -> fast -= inner_lr * grad   (SGD, fp32 fast weights)
    loss(query) at fast -> grad(fast) = first-order meta-gradient -> all-reduce over ranks
    theta_meta <- AdamW step (linear warmup, linear decay)
```

* `training_step` was split into `_get_training_inputs` + `training_step_from_inputs` so the inner
  loop reuses one VAE encode and only re-samples the flow-matching noise level / noise.
* `loss_mode = "action"` uses the flow-matching **action** loss only (ANIL-style; the vision
  loss reaches the action heads only through cross-attention). `"total"` uses the model's total loss.
* The backbone is a full bf16 replica on every rank without FSDP (`parallelism.enable_inference_mode=True`,
  `data_parallel_shard_degree=1`): the adapter must write fast weights into one row in place, which
  DTensor-sharded parameters do not allow. Only `theta_meta` gradients (~0.5M floats) cross ranks.
  With activation checkpointing a 40-window support batch fits comfortably on a 48 GB GPU.
* `theta_meta` starts from Cosmos3's fresh action-head init (`init_source="fresh"`), i.e. the same
  distribution the LIBERO baseline starts from; `checkpoint_row` / `checkpoint_mean` start from the
  mid-trained rows of the meta embodiments instead.
* Outputs in `$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_meta/fewshot_meta/<name>/`: `meta_action_init.pt`
  (latest `theta_meta`), `meta_action_init_iter_XXXXXX.pt` snapshots, `meta_state_latest.pt`
  (auto-resume), `meta_train_log.jsonl`, W&B. Watch `query_loss` (post-adaptation) vs
  `query_loss_zero_shot` (before adaptation) and `support_loss_first -> support_loss_last`: the gap is
  the few-shot adaptation the meta-init is buying; tune `inner_lr` so the support loss drops within
  `inner_steps` without diverging.

Launch (site SLURM wrapper `sr <gpus> <vram>`; all env vars from `~/.zshrc`):

```bash
cd ~/project/cosmos_hs07/packages/cosmos3
export ROBOT_FEWSHOT_ROOT=$COSMOS_STORAGE/data/robot_fewshot
export BASE_CHECKPOINT_PATH=$COSMOS_STORAGE/checkpoints/Cosmos3-Edge
# smoke: 2 GPUs, 5 iterations, no W&B
EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 job.wandb_mode=disabled" NPROC_PER_NODE=2 sr 2 48 examples/launch_meta_action_fewshot_edge.sh
# full run: 4 GPUs (4 meta-episodes per iteration)
NPROC_PER_NODE=4 sr 4 48 examples/launch_meta_action_fewshot_edge.sh
```

Every knob is in `[custom.meta]` of the TOML (`k_shot`, `q_query`, `windows_per_demo`, `inner_steps`,
`inner_lr`, `inner_optimizer`, `meta_lr`, `loss_mode`, `scratch_domain_id`, `init_source`, ...);
Hydra keys still work after `--` (e.g. `trainer.max_iter=3000`).

## 4. Downstream: LIBERO-10 few-shot post-training

The baseline (`action_policy_libero_10_edge.toml`, cosmos_hs04) skips the action heads when loading
the base checkpoint and trains them from a fresh init. **Reusing that recipe unchanged after
meta-training would throw `theta_meta` away** — hence `checkpoint.meta_action_init_*`:

```toml
[checkpoint]
load_path = "${oc.env:BASE_CHECKPOINT_PATH}"
meta_action_init_path      = "${oc.env:META_ACTION_INIT_PATH}"   # <meta run>/meta_action_init.pt
meta_action_init_domain_id = 5                                  # EMBODIMENT_TO_DOMAIN_ID["libero"]
```

`DistributedCheckpointer.load` applies it right after the warm-start model load
(`checkpoint/meta_action_init.py`): the LIBERO row of `action2llm` / `llm2action` and the shared
`action_modality_embed` are overwritten in `net` and `net_ema` (FSDP2 DTensors are gathered and
re-distributed). Everything else — trainable set (`moe_gen`, `time_embedder`, `vae2llm`, `llm2vae`,
`k_norm_und_for_gen`, action heads with the 5× LR multiplier), schedule, the fixed 3-demos-per-task
subset `libero_10_3ep_per_task_seed42.json` — is identical to the baseline, so the comparison isolates
the initialization. To make the downstream adaptation as restrictive as the meta stage, add
`[optimizer] keys_to_select = ["action2llm", "llm2action", "action_modality_embed"]` to *both* TOMLs.

```bash
export LIBERO_ROOT=$COSMOS_STORAGE/data/LIBERO_LeRobot_v3/libero_10
export META_ACTION_INIT_PATH=$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_meta/fewshot_meta/edge_meta_fomaml_k5q5_w8_seed42/meta_action_init.pt
NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_libero_10_edge.sh            # baseline (fresh init)
NPROC_PER_NODE=4 sr 4 48 examples/launch_sft_action_policy_libero_10_edge_metainit.sh   # ours (meta init)
```

## 5. Closed-loop evaluation

`examples/eval_libero_closed_loop.sh` wraps the two-process flow used so far (policy server in the
training env + LIBERO client in the `libero-eval` env):

```bash
RUN_ROOT=$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_libero/edge_libero10_fewshot/edge_libero10_3ep_per_task_sft_metainit_seed42 \
ITER=1500 OUT_DIR=$COSMOS_STORAGE/outputs/eval/edge_libero10_3ep_metainit_iter1500 NUM_TRIALS=50 \
  sr 1 48 examples/eval_libero_closed_loop.sh
```

Compare `summary.json` (`overall_success_rate`) of the baseline and the meta-init run at the same
iterations (500 / 1000 / 1500 / 2000).

## 6. Environment notes

* Always run from `packages/cosmos3` with `PYTHONPATH=.` (the `cosmos3-pt` env has cosmos_hs04
  installed in editable mode; the launchers set it).
* torchcodec (CUDA build) needs the NVIDIA NPP libraries from the pip `nvidia-*-cu13` wheels, which are
  not on the default loader path (`Could not load libtorchcodec ... libnppicc.so.13`). Previously this
  required exporting `LD_LIBRARY_PATH` by hand before every launch; `ActionBaseDataset.__init__` now
  pre-loads them (`data/generator/action/utils/video_decode_env.py`), so the meta trainer and the LIBERO
  recipes decode video without any shell setup. The manual export stays harmless.
* RoboMIND-UR FK needs `mujoco` (installed into `cosmos3-pt`; the reader now supports the 3.3.x
  `element.delete()` API as well as `MjSpec.delete`).
* CPU tests: `python -m pytest --noconftest -c /dev/null -p no:cacheprovider --rootdir=. \
  cosmos_framework/data/generator/action/yam_fk_test.py cosmos_framework/data/generator/action/meta/*_test.py`
  (the repo `conftest.py` needs pytest plugins that are not in the conda env).
* **`Too many open files` in the DataLoader workers → silent hang.** One meta-episode is ~1k small
  tensors; with PyTorch's default `file_descriptor` sharing strategy each one pins an fd in the worker and
  in the trainer, which overflows the compute nodes' soft `nofile` limit (1024). The worker's queue
  feeder then drops the item and the trainer blocks forever on `next(loader)` — visible as GPU 0 at 100 %
  (spinning in `all_reduce`) and the other ranks idle. `train_action_meta.py` now switches to the
  `file_system` strategy and raises the soft limit to the hard limit; the launcher also runs `ulimit -n`.
  `[custom.meta] loader_timeout_s` (default 1800) additionally turns any stalled worker into a hard error.
* **YAM decode tolerance.** LeRobot v3 repos store many episodes per mp4, so absolute timestamps reach
  thousands of seconds where lerobot's float32 tolerance check (`1e-4`) rejects perfectly aligned frames.
  `MolmoAct2YAMDataset` therefore defaults `tolerance_s` to half a frame period (`0.5 / fps`).
