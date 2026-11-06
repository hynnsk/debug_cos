# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_robocasa_edge`` -- Cosmos3-Edge RoboCasa365 target/atomic action-policy post-training (cosmos_hs11).

The LIBERO-10 Edge full-FT recipe (``action_policy_libero_edge``: same model config, trainable set incl.
``k_norm_und_for_gen``, optimizer / scheduler / EMA defaults, callbacks incl. ``ValLossBreakdownCallback``, checkpoint
skip list) on the RoboCasa365 data contract of ``action_policy_robocasa_nano``:

* the 18 ``target/atomic`` tasks (PandaOmron mobile manipulator), ~500 human demos/task (502-513 in the exports; the
  RoboCasa365 "target" split),
  read from ``ROBOCASA_ROOT`` = ``<...>/v30/target/atomic`` (v3.0 copies, see examples/setup_robocasa365.sh);
* action = 15D ``base_encoding="raw_armfirst"`` ``[eef_pos(3), eef_rot6d(6), gripper(1), base_motion(4), control_mode(1)]``:
  the Nano recipe's ``"raw"`` channels with the arm block FIRST, so dims 0..9 are exactly the 10D ``[pos, rot6d, gripper]``
  layout of every Reptile meta-training embodiment (fractal / bridge / RoboMIND-UR / RoboMIND-Franka, YAM = two of them)
  and of LIBERO; the meta-initialized action-head row therefore keeps its column semantics, and the base command + mode
  flag (mostly 0 / -1: only NavigateKitchen drives the base) land on dims 10..14, which YAM's right arm trained.
  ``action_normalization=None`` (arm deltas are normalised OSC commands and the base block a [-1, 1] command, so the raw
  channel scales already match);
* observation = ``camera_set="left_wrist"`` (``agentview_left`` | ``eye_in_hand`` at native 256x256 each -> 256x512, the
  LIBERO concat-view geometry) + the current EEF pose prepended as a clean conditioning token (``use_state=True``);
* ``chunk_length=16`` (17-frame windows at 20 fps) = the LIBERO / Reptile meta-training window contract, NOT the Nano
  recipe's 32: the Reptile theta was meta-trained on 17-frame windows, and the closed-loop client re-plans every
  ``ACTION_HORIZON=16`` steps (examples/eval_robocasa_closed_loop.sh).

Data fraction (RoboCasa365 Table 2 "Target only" rows): ``[dataloader_train].episodes_per_task`` = 50 (10 %), 150 (30 %)
or unset (100 % = every remaining train episode, ~500/task; the seeded 1 % val split = 5 demos/task is never trained on and is the
``dataloader_val`` set). Recipes: ``examples/toml/sft_config/action_policy_robocasa_edge_{baseinit,reptileinit}_{10,30,100}pct.toml``.
See docs/action_policy_robocasa_posttrain.md.
"""

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.val_loss_breakdown import ValLossBreakdownCallback
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_edge import (
    _action_policy_libero_edge_model_config,
)
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_robocasa_sft_dataset
from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import DEFAULT_ALL_ATOMIC_TASKS
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


def _robocasa_packing_loader(
    *,
    dataset_name: str,
    split: str,
    episodes_per_task: int | None,
    cfg_dropout_rate: float,
    episode_shuffle_seed: int,
    max_samples_per_batch: int,
    num_workers: int,
    prefetch_factor: int,
    use_image_augmentation: bool = False,
):
    """RoboCasa365 PackingDataLoader; train and val share everything but the split / fraction / dropout / seed.

    Window = 17 frames of the 256x512 [agentview_left | wrist] composite = 5 latent frames x 128 tokens = 640 vision
    tokens, exactly the LIBERO concat-view geometry. Token packing is off (``max_sequence_length=None``), so
    ``max_samples_per_batch`` binds: 128 windows/rank = global 512 on 4 ranks (the LIBERO Edge TOML runs 128 x 2 ranks
    = 256); ``max_num_tokens_after_packing=45056`` is only read by the cost model. Measured on the baseinit 10 % run:
    ~1.9 k tokens/window (640 gen + VLM/text/action), 238 k tokens/rank/iter, peak 29 GB/GPU, 55 s/iter."""
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
                        root="${oc.env:ROBOCASA_ROOT}",  # <ROBOCASA_DATA>/v30/target/atomic (<task>/<date>/lerobot shards)
                        task_names=DEFAULT_ALL_ATOMIC_TASKS,  # all 18 atomic tasks (NavigateKitchen included)
                        use_base_action=True,
                        base_encoding="raw_armfirst",  # 15D: arm(10) FIRST (LIBERO / meta layout), then base_motion(4) + control_mode
                        fps=20,
                        chunk_length=16,  # 17-frame windows = LIBERO / Reptile contract; eval ACTION_HORIZON=16
                        mode="wam",
                        viewpoint="concat_view",
                        camera_set="left_wrist",  # agentview_left | wrist, full-res 256x512
                        use_state=True,  # EEF proprioception -> prepended clean condition token
                        action_normalization=None,  # absolute state token: delta-stats do not apply
                        split=split,
                        split_val_ratio=0.01,  # 5 of the ~500 demos/task -> the val split; the rest = the 100 % train set
                        split_seed=42,
                        episodes_per_task=episodes_per_task,  # TOML [dataloader_train].episodes_per_task: 50 / 150 / None
                        episode_subset_seed=42,
                        iterable_shuffle=True,
                        episode_shuffle_seed=episode_shuffle_seed,
                        # Train-time random crop (95 %) + resize + ColorJitter, the DROID recipe / GR00T N1.5's RoboCasa365
                        # transform; off by default (the v1 runs had none), TOML [dataloader_train].use_image_augmentation.
                        use_image_augmentation=use_image_augmentation,
                        resolution=None,  # keep 256x512 horizontal concat un-squished
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
            # FusedAdam with fp32 master_weights + eps 1e-8 (as the LIBERO Edge recipe).
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},  # linear LR decay
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
            lr=5.0e-05,
            lr_multipliers={
                "action2llm": 5.0,  # fresh-head boost (baseinit); the reptileinit TOMLs set 1.0 (heads are meta-initialized)
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
                "lora_": 1.0,  # present so a TOML can override it; matches nothing while lora_enabled is False
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],  # smoke; the TOMLs set max_iter
            f_max=[1.0],
            f_min=[0.0],
            f_start=[1.0e-06],
            verbosity_interval=0,
            warm_up_steps=[0],  # smoke; the TOMLs set 200
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
            # The reptileinit TOMLs then overwrite row 30 with the meta heads (checkpoint.meta_action_init_*).
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
            episodes_per_task=None,  # 100 % (~500/task); the 10 / 30 % TOMLs set 50 / 150 via [dataloader_train].episodes_per_task
            cfg_dropout_rate=0.1,
            episode_shuffle_seed=42,
            max_samples_per_batch=128,  # binding (token packing off): 128 windows/rank x 4 ranks = global 512
            num_workers=4,
            prefetch_factor=4,
            use_image_augmentation=False,  # TOML [dataloader_train].use_image_augmentation = true to enable
        ),
        # Held-out validation: the seeded 1 % val split (5 demos/task = 90 episodes), no prompt dropout, its own shuffle
        # seed; every pass restarts at epoch 0 so successive val numbers use the same windows. Enabled from the TOML
        # ([trainer].run_validation / validation_iter / max_val_iter / run_validation_on_start).
        dataloader_val=_robocasa_packing_loader(
            dataset_name="action_robocasa_val",
            split="val",
            episodes_per_task=None,
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
