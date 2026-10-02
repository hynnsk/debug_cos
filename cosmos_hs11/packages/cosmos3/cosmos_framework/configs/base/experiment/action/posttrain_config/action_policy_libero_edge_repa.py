# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_libero_edge_repa`` -- Cosmos3-Edge LIBERO-10 action-policy SFT + REPA distillation loss.

cosmos_hs09_2: identical to this repo's ``action_policy_libero_edge`` (cosmos_hs08 recipe: full ``moe_gen``
post-training of the mid-trained Cosmos3-Edge on the fixed 30-demo LIBERO-10 subset, held-out validation, and
the cosmos_hs09 ``lora_`` LR-multiplier hook + ``[checkpoint].meta_action_init_*`` meta initialization) plus the
cosmos_hs10 representation-alignment term on the MoT generation pathway (ported from cosmos_hs10; the
``action_policy_libero_10_edge_metainit_repa_dinov2.toml`` recipe = hs10 v7 = frozen DINOv2 ViT-B/14 teacher):

    loss += repa.loss_weight * (1 - cos(proj(h_k[video tokens]), adapter(V-JEPA-2.1(frames))))
          + repa.relation_loss_weight * dist(R(proj(h_k)), R(adapter(V-JEPA-2.1(frames))))   # R = token-relation map

* ``proj``: the REPA 3-layer SiLU MLP (``repa.projector_type="mlp"``, default) or a single linear layer
  (``"linear"``, the ``*_v4.toml`` recipe).
* ``R(x) = normalize(x) normalize(x)^T`` per sample (pairwise cosine similarities among the predicted tokens of one
  video, spatial and temporal pairs alike); ``dist`` = squared (``"l2"``) or absolute (``"l1"``) entry-wise difference.
  VideoREPA-style relation distillation; the ``*_v5.toml`` recipe trains it alone (``loss_weight=0``,
  ``relation_loss_weight=5``). Both terms are always logged (``repa_loss``/``repa_cos_sim``, ``repa_rel_loss``).
* ``h_k``: hidden state of the *predicted* (noised) video tokens after MoT block ``repa.layer_index`` (8 of 28
  by default; 14 is the next candidate).
* teacher: frozen V-JEPA 2.1 ``ema_encoder`` (ViT-B/16 default, ViT-L/16 variant) on the 16 future frames of
  each camera view at its native 256x256 resolution (ImageNet normalization), i.e. ``8x16x16`` tokens per view.
* target: per view the teacher grid is brought onto that view's MoT token grid (``4x5x5``) by one of the
  ``repa.target_adapter`` variants (``avgpool`` / ``avgpool_conv`` / ``strided_conv``); the two views are then
  concatenated along width (``4x5x10``), exactly like the third-person | wrist canvas.

Data: the dataset additionally ships the un-resized uint8 frames as ``video_native`` (``keep_native_video``);
the model canvas / VAE path is untouched. New trainable params: ``net.repa_head.*`` (``repa_`` prefix ->
``keys_to_select`` / ``keys_to_skip_loading``). See docs/action_policy_libero_repa_vjepa.md.
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


def _action_policy_libero_edge_repa_model_config() -> dict:
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)

    # Action-policy training (same as action_policy_libero_edge)
    cfg["action_gen"] = True
    cfg["vision_gen"] = True
    cfg["max_action_dim"] = 64

    # V-JEPA 2.1 representation alignment. Every knob is TOML-overridable under [model.repa]
    # (schema: configs/toml_config/sft_config.py::RepaTomlConfig -> model.config.repa.*).
    cfg["repa"] = dict(
        enabled=True,
        loss_weight=0.5,
        loss_weight_warmup_steps=0,  # 0 = constant weight; cosmos_hs11 v3 = 200 (linear 0 -> loss_weight ramp)
        sigma_min=0.0,  # noise-level gate (hs11 v6/v7): REPA only for samples with sigma in [sigma_min, sigma_max]
        sigma_max=1.0,
        objective="token",  # "temporal_difference" (v8) | "spatial_normalized" (v10)
        spatial_norm_eps=1.0e-6,
        masked_ratio_min=0.4,  # objective="masked_prediction" (cosmos_hs12): tube-mask ratio range over target cells
        masked_ratio_max=0.7,
        masked_visible_weight=0.25,
        masked_max_samples=16,
        masked_warmup_steps=200,
        masked_seed=42,
        relation_loss_weight=0.0,  # VideoREPA-style token-relation term; v5 = 5.0 (with loss_weight 0.0)
        relation_distance="l2",  # "l2" (squared) | "l1" (absolute) entry-wise relation-map difference
        layer_index=8,  # output of MoT block 8 (of 28); next candidate: 14
        teacher="vjepa2_1_vit_base_384",  # ViT-B/16 default; "vjepa2_1_vit_large_384" = ViT-L/16
        teacher_checkpoint_path=None,  # $COSMOS_STORAGE/checkpoints/vjepa2_1/<release file>
        teacher_input_size=256,  # LIBERO native camera resolution == V-JEPA 2.1 RoPE base grid (16x16)
        teacher_num_frames=16,  # = chunk_length: the predicted frames 1..16
        teacher_batch_size=32,
        target_adapter="avgpool",  # variant 1; "avgpool_conv" = variant 2; "strided_conv" = variant 3
        target_adapter_kernel_size=3,
        target_adapter_depthwise=True,
        target_grid_thw=(4, 5, 5),  # per-view MoT token grid: 4 predicted latent frames x 5 x 5 (160-px content)
        projector_type="mlp",  # REPA MLP; "linear" = single Linear(hidden_size, D_t) (v4 recipe)
        projector_hidden_dim=2048,  # MLP hidden width (unused for "linear")
        num_views=2,  # third-person | wrist, concatenated along width
        native_video_key="video_native",
    )
    # Optional anti-collapse regularizer over MoT visual tokens only. v9 enables this from TOML. It introduces no
    # parameters and therefore needs no optimizer/checkpoint allowlist entry.
    cfg["sigreg"] = dict(
        enabled=False,
        loss_weight=0.1,
        layer_index=8,
        num_slices=256,
        num_points=17,
        integration_max=5.0,
        slice_batch_size=64,
        seed=0,
    )
    return cfg


def _libero_packing_loader(
    *,
    dataset_name: str,
    episode_subset_path: str | None,
    cfg_dropout_rate: float,
    episode_shuffle_seed: int,
    max_samples_per_batch: int,
    num_workers: int,
    prefetch_factor: int,
):
    """LIBERO-10 PackingDataLoader (+ native frames for the REPA teacher); train and val share everything but
    the episode subset / dropout / seed."""
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
                        root="${oc.env:LIBERO_ROOT}",
                        fps=20,  # metadata only (FPS-agnostic loader reads native fps from info.json)
                        chunk_length=16,
                        image_size=256,  # concat_view -> 256x512 (native copy kept for the teacher)
                        mode="wam",
                        camera_mode="concat_view",
                        action_space="frame_wise_relative",
                        rotation_space="6d",
                        pose_coordinate_frame="native",
                        action_normalization="quantile_rot",
                        val_ratio=0.01,
                        episode_subset_path=episode_subset_path,
                        iterable_shuffle=True,
                        episode_shuffle_seed=episode_shuffle_seed,
                        resolution=None,
                        max_action_dim="${model.config.max_action_dim}",
                        cfg_dropout_rate=cfg_dropout_rate,
                        format_prompt_as_json=True,
                        tokenizer_config="${model.config.vlm_config.tokenizer}",
                        keep_native_video=True,  # ships uint8 [3,17,256,512] as data_batch["video_native"]
                    ),
                ),
            ),
        ),
    )


action_policy_libero_edge_repa = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
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
            name="action_policy_libero_edge_repa",
            wandb_mode="disabled",
        ),
        model=dict(
            config=_action_policy_libero_edge_repa_model_config(),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            # Generation + action heads (hs08 recipe) + the REPA projector / target adapter.
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "k_norm_und_for_gen",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "repa_",
            ],
            lr=5.0e-05,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
                # cosmos_hs09 hook (Hydra cannot add new dict keys from a TOML): 1.0 = no effect; the meta-init
                # full-FT recipes set it to 0.0 to freeze the meta-learned LoRA delta (== merged into moe_gen).
                # Matches nothing when lora_enabled is False.
                "lora_": 1.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],
            f_max=[1.0],
            f_min=[0.0],
            f_start=[1.0e-06],
            verbosity_interval=0,
            warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=100,
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
                # val/loss_total, val/flow_matching_loss_{action,vision}, val/repa_loss, val/repa_cos_sim, val/repa_rel_loss
                val_loss_breakdown=L(ValLossBreakdownCallback)(
                    keys=[
                        "flow_matching_loss_action",
                        "flow_matching_loss_vision",
                        "repa_loss",
                        "repa_cos_sim",
                        "repa_rel_loss",
                        "repa_weight",
                        "repa_weighted_loss",
                        "repa_sigma_frac",
                        "repa_cos_sim_centered",
                        "repa_cos_sim_transition",
                        "repa_cos_sim_spatial_norm",
                        "sigreg_loss",
                        "jepa_loss",
                        "jepa_masked_loss",
                        "jepa_visible_loss",
                        "jepa_weighted_loss",
                        "jepa_mask_fraction",
                        "jepa_centered_cos",
                        "jepa_pred_std",
                        "jepa_weight",
                    ],
                ),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Skip net_ema (EMA warm-starts from net), the action heads (fresh init as in hs08) and the REPA head
            # (absent from the base checkpoint). The meta-init TOML additionally lists "lora_" (the base DCP has no
            # adapters) and re-states this list, so keep "repa_" there as well.
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
                "repa_",
            ],
            load_ema_to_reg=False,
            load_path="???",
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,
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
            episode_subset_path=None,
            cfg_dropout_rate=0.1,
            episode_shuffle_seed=42,
            max_samples_per_batch=128,
            num_workers=4,
            prefetch_factor=4,
        ),
        dataloader_val=_libero_packing_loader(
            dataset_name="action_libero_val",
            episode_subset_path="libero_10_val_5ep_per_task_seed42_excl3ep.json",
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


for _item in [action_policy_libero_edge_repa]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
