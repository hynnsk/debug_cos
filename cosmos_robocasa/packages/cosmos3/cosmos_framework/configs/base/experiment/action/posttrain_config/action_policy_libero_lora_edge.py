# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_libero_lora_edge`` -- Cosmos3-Edge LIBERO-10 few-shot post-training in the LoRA regime (cosmos_hs09).

Same data / loss / schedule as ``action_policy_libero_edge`` (the full-``moe_gen`` Cosmos recipe), but the
trainable set is exactly the theta_meta groups of ``action_fewshot_meta_lora_edge``:

* LoRA adapters (rank 32, alpha 64) on the generation tower ``*_moe_gen`` q/k/v/o + ``mlp_moe_gen`` up/down
  in every layer (~33M params),
* the action heads ``action2llm`` / ``llm2action`` / ``action_modality_embed`` (fresh; 0.27M trainable row),
* ``time_embedder`` (4.7M, mid-trained).

Everything else -- the understanding tower, the ``moe_gen`` base weights (1.4B), ``vae2llm`` / ``llm2vae``,
``k_norm_und_for_gen`` -- is frozen. One learning rate (1e-4, no per-module multipliers): every trainable
module is an adapter-sized object, and the recipe's 5x head multiplier existed only because the heads were
the single fresh module of a full fine-tune.

* **Baseline**: LoRA ``B = 0`` (identity adapter), fresh heads, mid-trained ``time_embedder`` -- i.e. the
  mid-trained Cosmos3-Edge adapted to LIBERO-10 with LoRA.
* **Ours**: the same parameters initialized from ``meta_action_init.pt`` (``checkpoint.meta_action_init_path``).
  The two TOMLs differ in exactly the ``[checkpoint].meta_action_init_*`` lines, and the init now covers
  100% of the trainable parameters (cosmos_hs07 covered 0.02%).

Validation: 50 held-out LIBERO-10 demos (``libero_10_val_5ep_per_task_seed42_excl3ep.json``, disjoint from
the 3-demo training subset) every ``trainer.validation_iter`` steps and at iteration 0, logged as ``val/*``
(``ValLossBreakdownCallback``: total / action / vision flow-matching loss).
See docs/action_fewshot_meta_lora.md.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.val_loss_breakdown import ValLossBreakdownCallback
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_libero_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

# Generation-tower linears that get a LoRA adapter. Path-qualified ``mlp_moe_gen.*`` entries are needed
# because the understanding tower's ``mlp.up_proj`` shares the leaf name (see utils/generator/lora.py).
LORA_TARGET_MODULES = (
    "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen,mlp_moe_gen.up_proj,mlp_moe_gen.down_proj"
)
LORA_RANK = 32
LORA_ALPHA = 64

# Exactly the theta_meta groups of action_fewshot_meta_lora_edge (substring match on parameter names).
LORA_RECIPE_TRAINABLE_KEYS = ["lora_", "action2llm", "llm2action", "action_modality_embed", "time_embedder"]

TRAIN_EPISODE_SUBSET = "libero_10_3ep_per_task_seed42.json"  # 3 demos/task = 30 episodes
VAL_EPISODE_SUBSET = "libero_10_val_5ep_per_task_seed42_excl3ep.json"  # 5 held-out demos/task = 50 episodes


def _action_policy_libero_lora_edge_model_config() -> dict:
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
    cfg["action_gen"] = True
    cfg["vision_gen"] = True
    cfg["max_action_dim"] = 64
    # LoRA on the generation tower; adapters are injected before FSDP wrap and initialized A~kaiming, B=0.
    cfg["lora_enabled"] = True
    cfg["lora_rank"] = LORA_RANK
    cfg["lora_alpha"] = LORA_ALPHA
    cfg["lora_target_modules"] = LORA_TARGET_MODULES
    # LoRA is trained TOGETHER with the action heads and time_embedder: do not let the injector freeze
    # them (the optimizer only sees parameters that already require grad; keys_to_select does the selection).
    cfg["lora_freeze_base"] = False
    return cfg


def _libero_packing_loader(
    *,
    dataset_name: str,
    episode_subset_path: str,
    cfg_dropout_rate: float,
    episode_shuffle_seed: int,
    max_samples_per_batch: int,
    num_workers: int,
    prefetch_factor: int,
):
    """LIBERO-10 PackingDataLoader (train and val share everything but the episode subset / dropout / seed)."""
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
            datasets=dict(
                libero=dict(
                    ratio=1,
                    dataset=L(get_action_libero_sft_dataset)(
                        root="${oc.env:LIBERO_ROOT}",  # local LeRobot dir of the libero_10 suite (20 FPS LIBERO_LeRobot_v3)
                        fps=20,
                        chunk_length=16,
                        image_size=256,  # concat_view -> 256x512
                        mode="wam",
                        camera_mode="concat_view",
                        action_space="frame_wise_relative",
                        rotation_space="6d",
                        pose_coordinate_frame="native",
                        action_normalization="quantile_rot",
                        val_ratio=0.01,  # unused with a fixed episode subset (the subset IS the split)
                        episode_subset_path=episode_subset_path,
                        iterable_shuffle=True,
                        episode_shuffle_seed=episode_shuffle_seed,
                        resolution=None,
                        max_action_dim="${model.config.max_action_dim}",
                        cfg_dropout_rate=cfg_dropout_rate,
                        format_prompt_as_json=True,
                        tokenizer_config="${model.config.vlm_config.tokenizer}",
                    ),
                ),
            ),
        ),
    )


action_policy_libero_lora_edge = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            # FusedAdam with fp32 master_weights + eps 1e-8 (bf16 params + eps 1e-6 diverged on the action loss).
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
            name="action_policy_libero_lora_edge",
            wandb_mode="disabled",
        ),
        model=dict(
            config=_action_policy_libero_lora_edge_model_config(),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,  # popped by build_optimizer for FusedAdam (fused by construction)
            keys_to_select=list(LORA_RECIPE_TRAINABLE_KEYS),
            # One adapter-scale LR for LoRA + heads + time_embedder. No 5x head multiplier: with a
            # meta-learned (or any pretrained) head init a 5x LR only erases the init faster, and in this
            # regime every trainable module is small. Same for baseline and ours.
            lr=1.0e-04,
            lr_multipliers={},
            optimizer_type="FusedAdam",
            weight_decay=0.0,  # LoRA convention (vision_sft_super); the full-moe_gen recipe used 0.05
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],  # smoke; the run TOML sets 16000
            f_max=[1.0],
            f_min=[0.0],
            f_start=[1.0e-06],
            verbosity_interval=0,
            warm_up_steps=[0],  # smoke; the run TOML sets 500
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=100,  # smoke; the run TOML sets 2000
            # Held-out validation (dataloader_val below): every validation_iter steps + at iteration 0, where
            # it measures the initialization itself (baseline fresh init vs meta init) before any update.
            max_val_iter=16,
            run_validation=True,
            run_validation_on_start=True,
            validation_iter=100,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
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
                # val/loss_total, val/flow_matching_loss_action, val/flow_matching_loss_vision
                val_loss_breakdown=L(ValLossBreakdownCallback)(),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Skip: net_ema (warm-started from net), the LoRA adapters (absent from the base DCP; they keep
            # their fresh A~kaiming / B=0 init) and the action heads (fresh; the public base has no
            # LIBERO-trained rows). `meta_action_init_path` (ours) overwrites heads + LoRA + time_embedder
            # right after this load. `action_pos_embed` of the old recipe does not exist on Edge.
            keys_to_skip_loading=[
                "net_ema.",
                "lora_",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Edge DCP dir; supply via TOML/env
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
        dataloader_train=_libero_packing_loader(
            dataset_name="action_libero",
            episode_subset_path=TRAIN_EPISODE_SUBSET,
            cfg_dropout_rate=0.1,
            episode_shuffle_seed=42,
            max_samples_per_batch=128,  # global = 128 x 4 ranks = 512 windows / step
            num_workers=4,
            prefetch_factor=4,
        ),
        # Held-out demos, no prompt dropout, its own shuffle seed; iterated from the start on every
        # validation pass (ActionIterableShuffleDataset restarts at epoch 0), so successive val numbers
        # are computed on the same windows.
        dataloader_val=_libero_packing_loader(
            dataset_name="action_libero_val",
            episode_subset_path=VAL_EPISODE_SUBSET,
            cfg_dropout_rate=0.0,
            episode_shuffle_seed=123,
            max_samples_per_batch=128,  # x max_val_iter(16) x 4 ranks ~= the whole 50-episode val set
            num_workers=2,
            prefetch_factor=2,
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


for _item in [action_policy_libero_lora_edge]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
