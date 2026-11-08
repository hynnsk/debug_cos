# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_robocasa_edge`` -- Cosmos3-Edge post-training on the ORIGINAL RoboCasa benchmark (cosmos_robocasa).

Data = the Cosmos Policy export of RoboCasa (``nvidia/RoboCasa-Cosmos-Policy``, ``success_only/``): the 24 atomic kitchen
tasks (PandaMobile, the base never moves) x the ~50 human demos per task (RSS 2024 "human-50" setting), replayed at
224x224 with no-op filtering and success trimming (Cosmos Policy, ICLR 2026, Sec. 5.1 / A.2.3), converted to LeRobot v3
shards by ``cosmos_framework/scripts/convert_robocasa_cosmos_policy_to_lerobot_v30.py`` (``ROBOCASA_ROOT/<Task>/<date>/lerobot``).

The data contract follows the Cosmos Policy RoboCasa recipe wherever the Cosmos3 action stack allows it:

* action = the 7D arm command ``[eef_pos(3), eef_rot_axisangle(3), gripper(1)]`` (Cosmos Policy drops the 5 mobile-base
  channels too) -> 10D ``[pos(3), rot6d(6), gripper(1)]`` (``use_base_action=False``), the layout of LIBERO and of every
  Reptile meta-training embodiment, so a meta-initialized action-head row keeps its column semantics;
  ``action_normalization="quantile_rot"`` (q01/q99 -> [-1, 1], the LIBERO scheme; Cosmos Policy: min/max -> [-1, 1]) with
  the bundled ``robocasa_cosmos_policy_frame_wise_relative_rot6d.json`` stats (computed by the converter);
* proprio = the current eef pose + gripper opening prepended as a clean conditioning token (``use_state=True``; Cosmos
  Policy ``use_proprio``), min/max-normalized by its own ``state`` statistics (``state_normalization="minmax"``);
* observation = ``camera_set="left_wrist"``: ``agentview_left | eye_in_hand`` at native 224x224 each (224x448 -> the
  256-tier 192x320 bucket of the transform), the LIBERO / Reptile canvas. Cosmos Policy feeds all three 224x224 cameras
  as separate latent frames; ``camera_set="wrist_lr"`` tiles wrist + left/right (half size) instead -- override both
  loader nodes (``dataloader_{train,val}.dataloader.datasets.robocasa.dataset.camera_set=wrist_lr``) to try it;
* ``chunk_length=16`` (17-frame windows at 20 fps) = the LIBERO / Reptile meta-training window contract (Cosmos Policy
  predicts 32 and executes 16; the Cosmos3 closed-loop client re-plans every 16 steps);
* train-time augmentation = Cosmos Policy's "stronger" aug: random crop to 90 % of the area + resize back, random
  rotation in [-5, 5] deg, ColorJitter 0.3 / 0.4 / 0.5 / 0.05, one draw per window applied to every frame of every view;
* no prompt dropout (``cfg_dropout_rate=0.0``; Cosmos Policy: "we don't want any text dropout; otherwise, the model may
  fail to follow language") -> evaluate with guidance 1.0 (no CFG);
* every success demo is trained on (``split_val_ratio=0.0``, like Cosmos Policy: no validation split); the val loader
  holds a seeded 5 % IN-SAMPLE subset (~3 demos/task) for loss monitoring only -- it is NOT held out.

Optimizer = Cosmos Policy's (the Cosmos-Predict2 defaults): AdamW lr 1e-4, betas (0.9, 0.99), eps 1e-8, weight decay 0.1,
grad clip 1.0; its schedule (warmup, linear decay to 0.3x, then 0.06x constant), batch and iteration budget live in the
TOMLs (``examples/toml/sft_config/action_policy_robocasa_edge_{baseinit,reptileinit}.toml``). Trainable set, EMA,
callbacks and the checkpoint skip list = the LIBERO Edge full-FT recipe (= the Reptile theta).
See docs/action_policy_robocasa_posttrain.md.
"""

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.val_loss_breakdown import ValLossBreakdownCallback
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_edge import (
    _action_policy_libero_edge_model_config,
)
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_robocasa_sft_dataset
from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import (
    DEFAULT_ROBOCASA_COSMOS_POLICY_TASKS,
)
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

# Cosmos Policy RoboCasa data settings that the Cosmos3 reader reproduces (see the module docstring).
ROBOCASA_CAMERA_SET = "left_wrist"
ROBOCASA_ACTION_NORMALIZATION = "quantile_rot"
ROBOCASA_STATE_NORMALIZATION = "minmax"
ROBOCASA_AUGMENT_CROP_SCALE = 0.95  # side 0.95 ~ 90 % of the area (Cosmos Policy RandomResizedCrop scale 0.9)
ROBOCASA_AUGMENT_ROTATION_DEG = 5.0
ROBOCASA_AUGMENT_HUE = 0.05
ROBOCASA_CFG_DROPOUT_RATE = 0.0
ROBOCASA_VAL_RATIO_IN_SAMPLE = 0.05


def _robocasa_packing_loader(
    *,
    dataset_name: str,
    split: str,
    split_val_ratio: float,
    cfg_dropout_rate: float,
    episode_shuffle_seed: int,
    max_samples_per_batch: int,
    num_workers: int,
    prefetch_factor: int,
    use_image_augmentation: bool = False,
    camera_set: str = ROBOCASA_CAMERA_SET,
):
    """RoboCasa PackingDataLoader; train and val share everything but the split / dropout / augmentation / seed.

    Window = 17 frames of the 224x448 [agentview_left | wrist] composite -> 192x320 bucket = 5 latent frames x 96 tokens
    of vision, the LIBERO concat-view geometry. Token packing is off (``max_sequence_length=None``), so
    ``max_samples_per_batch`` binds: 128 windows/rank = global 512 on 4 ranks; ``max_num_tokens_after_packing=45056`` is
    only read by the cost model. The RoboCasa365 recipe of cosmos_hs11 measured ~1.9 k tokens/window, 29 GB peak/GPU and
    55 s/iter at this geometry (34.5 s of it VAE encoding)."""
    return L(PackingDataLoader)(
        audio_sample_rate=48000,
        dataset_name=dataset_name,
        max_samples_per_batch=max_samples_per_batch,
        max_sequence_length=None,  # None disables token packing (TOML can't express null)
        patch_spatial=2,
        sound_latent_fps=0,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        dataloader=L(RankPartitionedDataLoader)(
            batch_size=1,
            in_order=False,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            sampler=None,
            # Shuffling is handled by the dataset (iterable_shuffle=True below): rank x worker-sharded,
            # episode-order-shuffled, sequential-within-episode (same as the LIBERO recipe).
            datasets=dict(
                robocasa=dict(
                    ratio=1,
                    dataset=L(get_action_robocasa_sft_dataset)(
                        root="${oc.env:ROBOCASA_ROOT}",  # <...>/robocasa_cosmos_policy_lerobot_v3 (<Task>/<date>/lerobot shards)
                        task_names=DEFAULT_ROBOCASA_COSMOS_POLICY_TASKS,  # the 24 original atomic tasks
                        use_base_action=False,  # 10D [pos, rot6d, gripper]: the fixed-base tasks never move the base
                        base_encoding="raw_armfirst",  # inert with use_base_action=False
                        fps=20,
                        chunk_length=16,  # 17-frame windows = LIBERO / Reptile contract; eval re-plans every 16 steps
                        mode="wam",
                        viewpoint="concat_view",
                        camera_set=camera_set,  # agentview_left | wrist, native 224x448 (or "wrist_lr" for all 3 cameras)
                        use_state=True,  # eef proprioception -> prepended clean condition token (Cosmos Policy use_proprio)
                        action_normalization=ROBOCASA_ACTION_NORMALIZATION,  # q01/q99 -> [-1, 1] (bundled stats)
                        state_normalization=ROBOCASA_STATE_NORMALIZATION,  # min/max -> [-1, 1] for the state token
                        action_stats_path=None,  # default: normalizer_stats/robocasa_cosmos_policy_frame_wise_relative_rot6d.json
                        split=split,
                        split_val_ratio=split_val_ratio,  # train: 0.0 = every success demo; val: 5 % in-sample monitor
                        split_seed=42,
                        episodes_per_task=None,  # all demos (the RoboCasa "human-50" setting)
                        episode_subset_seed=42,
                        iterable_shuffle=True,
                        episode_shuffle_seed=episode_shuffle_seed,
                        # Cosmos Policy "stronger" augmentation (train only; the val loader ignores the flag).
                        use_image_augmentation=use_image_augmentation,
                        augment_crop_scale=ROBOCASA_AUGMENT_CROP_SCALE,
                        augment_rotation_deg=ROBOCASA_AUGMENT_ROTATION_DEG,
                        augment_hue=ROBOCASA_AUGMENT_HUE,
                        resolution=None,  # auto tier: 224x448 -> "256" -> 192x320 bucket (aspect-preserving + reflection pad)
                        max_action_dim="${model.config.max_action_dim}",
                        cfg_dropout_rate=cfg_dropout_rate,
                        format_prompt_as_json=True,
                        tokenizer_config="${model.config.vlm_config.tokenizer}",
                    ),
                ),
            ),
        ),
    )


action_policy_robocasa_edge = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            # FusedAdam with fp32 master_weights (as the LIBERO Edge recipe); the values below are Cosmos Policy's.
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},  # Cosmos Policy: linear decay to 0.3x, then 0.06x (2 cycles, TOML)
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="action_sft",
            name="action_policy_robocasa_edge",
            wandb_mode="disabled",
        ),
        model=dict(
            # Identical to the LIBERO Edge recipe (action_gen + vision_gen, max_action_dim 64, language_ranking off).
            config=_action_policy_libero_edge_model_config(),
        ),
        optimizer=dict(
            # Cosmos Policy / Cosmos-Predict2 optimizer defaults (cosmos_policy/_src/predict2/configs/common/defaults/optimizer.py):
            # AdamW lr 1e-4, weight_decay 0.1, betas [0.9, 0.99], eps 1e-8. The LIBERO Edge recipe used 5e-5 / 0.05.
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,  # popped by build_optimizer for FusedAdam (fused by construction)
            # Train the generation + action heads: the LIBERO Edge trainable set (= the Reptile theta).
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "k_norm_und_for_gen",  # Edge-only
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=1.0e-04,
            lr_multipliers={
                "action2llm": 5.0,  # fresh-head boost (baseinit); the reptileinit TOML sets 1.0 (heads are meta-initialized)
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
                "lora_": 1.0,  # present so a TOML can override it; matches nothing while lora_enabled is False
            },
            optimizer_type="FusedAdam",
            weight_decay=0.1,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],  # smoke; the TOMLs set the Cosmos Policy 2-cycle schedule
            f_max=[1.0],
            f_min=[0.0],
            f_start=[1.0e-06],
            verbosity_interval=0,
            warm_up_steps=[0],  # smoke; the TOMLs set the warmup
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=100,  # smoke
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
                # val/loss_total, val/flow_matching_loss_action, val/flow_matching_loss_vision (when run_validation)
                val_loss_breakdown=L(ValLossBreakdownCallback)(),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Skip net_ema (EMA warm-starts from net, see dcp.py) and the action heads, so they init fresh from the base
            # (the mid-trained Cosmos3-Edge has no RoboCasa-trained action-head row; domain row 30 = "robocasa").
            # The reptileinit TOML then overwrites row 30 with the meta heads (checkpoint.meta_action_init_*).
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Edge DCP dir (baseinit) or a Reptile iteration dir (reptileinit); set in the TOML
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,  # base init: tolerate key set differences
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=_robocasa_packing_loader(
            dataset_name="action_robocasa",
            split="train",
            split_val_ratio=0.0,  # every success demo (Cosmos Policy trains on all of them; no held-out split)
            cfg_dropout_rate=ROBOCASA_CFG_DROPOUT_RATE,
            episode_shuffle_seed=42,
            max_samples_per_batch=128,  # binding (token packing off): 128 windows/rank x 4 ranks = global 512
            num_workers=4,
            prefetch_factor=4,
            use_image_augmentation=True,  # TOML [dataloader_train].use_image_augmentation = false to disable
        ),
        # Loss monitor only: a seeded 5 % subset (~3 demos/task) of the SAME demos, no prompt dropout, no augmentation,
        # its own shuffle seed; every pass restarts at epoch 0 so successive numbers use the same windows. Off by default
        # ([trainer].run_validation); it is in-sample, not a generalisation measure.
        dataloader_val=_robocasa_packing_loader(
            dataset_name="action_robocasa_val",
            split="val",
            split_val_ratio=ROBOCASA_VAL_RATIO_IN_SAMPLE,
            cfg_dropout_rate=0.0,
            episode_shuffle_seed=123,
            max_samples_per_batch=128,
            num_workers=2,
            prefetch_factor=2,
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


for _item in [action_policy_robocasa_edge]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
