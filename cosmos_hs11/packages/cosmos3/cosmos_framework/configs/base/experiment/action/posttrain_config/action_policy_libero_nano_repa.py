# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_libero_nano_repa`` -- Cosmos3-Nano LIBERO-10 few-shot action-policy SFT + REPA loss.

cosmos_hs11: ported from cosmos_hs10 (2026-09-30) so the Nano Reptile-init post-training recipes can add the REPA
distillation term (``action_policy_libero_10_nano_reptileinit_v2.toml`` = the plain Nano reptileinit recipe + frozen
DINOv2 ViT-L/14 teacher on the LAST MoT block, 36). The Reptile warm start itself is entirely ``[checkpoint]``-driven
(``load_path`` = Reptile DCP, ``meta_action_init_*``), so this experiment only has to (a) add ``repa_`` to
``keys_to_select`` / ``keys_to_skip_loading`` and (b) ship the native frames for the teacher; the model config carries
the hs11 ``RepaConfig`` defaults (``loss_weight_warmup_steps``, ``masked_*``, ``target_subgrid_thw``).

Nano counterpart of ``action_policy_libero_edge_repa`` (cosmos_hs10): the cosmos_hs08 few-shot Nano recipe
(Cosmos3-Nano = Qwen3-VL-8B MoT, hidden 4096, 36 decoder blocks; fixed 30-demo LIBERO-10 subset, held-out
validation, FSDP shard 8 x 32 windows/rank on one 8 x 48 GB node, torch.compile OFF because Inductor's fused
RMSNorm-backward kernel for the 4096-wide language region exceeds Ampere's shared memory) plus the representation
alignment term on the MoT generation pathway:

    loss += repa.loss_weight * (1 - cos(proj(h_k[predicted video tokens]), adapter(teacher(frames))))

Differences from the Edge REPA experiment, all TOML-overridable under ``[model.repa]``:
  * teacher defaults to the ViT-L tier (``vjepa2_1_vit_large_384``, D=1024; DINOv2 = ``dinov2_vitl14``) to match the
    ~5x larger student;
  * ``layer_index`` defaults to 10: the Edge depth 8 of 28 blocks (~29%) mapped onto Nano's 36 blocks.
The projector reads Nano's 4096-wide residual stream (``hidden_size`` is taken from the backbone config), the
video-token grid is unchanged (same VAE / patchify / 192x320 canvas -> 5 x 5 x 10, i.e. 4 x 5 x 5 per view).
The dataset additionally ships the un-resized uint8 frames as ``video_native`` (``keep_native_video``). New
trainable params: ``net.repa_head.*`` (``repa_`` prefix -> ``keys_to_select`` / ``keys_to_skip_loading``).
See docs/action_policy_libero_repa_vjepa.md (section 8).
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.val_loss_breakdown import ValLossBreakdownCallback
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_libero_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


def _action_policy_libero_nano_repa_model_config() -> dict:
    """LIBERO model config: capped packed tokens, selective activation
    checkpointing, fresh diffusion-expert init, 10x vision flow-matching loss.
    Keep ``encode_exact_durations=[17, 61, 73]`` to match the Cosmos3-Nano base."""
    cfg = copy.deepcopy(NANO_MODEL_CONFIG)  # action_gen=True, max_action_dim=64
    # Cap the packed sequence. Uncapped (-1) + a large max_samples_per_batch packs
    # one very long sequence and OOMs even on H200; 74000 keeps the GA-validated bound.
    cfg["max_num_tokens_after_packing"] = 74000
    cfg["activation_checkpointing"]["mode"] = "selective"
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["rectified_flow_training_config"]["loss_scale"] = 10.0
    cfg["rectified_flow_training_config"]["image_loss_scale"] = None
    cfg["tokenizer"]["encode_exact_durations"] = [17, 61, 73]  # match Cosmos3 base + reference SFT (do NOT reduce)
    cfg["action_gen"] = True
    cfg["vision_gen"] = True
    cfg["max_action_dim"] = 64

    # Representation alignment. Every knob is TOML-overridable under [model.repa]
    # (schema: configs/toml_config/sft_config.py::RepaTomlConfig -> model.config.repa.*).
    cfg["repa"] = dict(
        enabled=True,
        loss_weight=5.0,
        loss_weight_warmup_steps=0,  # hs11 v3-style linear ramp of the cosine weight; 0 = constant
        sigma_min=0.0,  # noise-level gate (hs11 v6/v7): REPA only for samples with sigma in [sigma_min, sigma_max]
        sigma_max=1.0,
        objective="token",  # "temporal_difference" (v8) | "spatial_normalized" (v10) | "masked_prediction" (hs12)
        # masked_prediction knobs (unused unless objective="masked_prediction")
        masked_ratio_min=0.4,
        masked_ratio_max=0.7,
        masked_visible_weight=0.25,
        masked_max_samples=16,
        masked_warmup_steps=200,
        masked_seed=42,
        spatial_norm_eps=1.0e-6,
        relation_loss_weight=0.0,  # VideoREPA-style token-relation term; 0 = monitor only
        relation_distance="l2",
        relation_margin=0.0,  # VideoREPA TRD margin; 0 = plain distance
        layer_index=10,  # output of MoT block 10 (of 36) ~= the Edge depth 8 of 28; the TOML may change it
        teacher="vjepa2_1_vit_large_384",  # ViT-L/16 (D=1024) for the Nano student; "dinov2_vitl14" = DINOv2 ViT-L/14
        teacher_checkpoint_path=None,  # $COSMOS_STORAGE/checkpoints/vjepa2_1/<release file> | HF cache for DINOv2
        teacher_input_size=256,  # V-JEPA: 256 (RoPE base grid); DINOv2: 224 (16x16 patches of 14 px)
        teacher_num_frames=16,  # = chunk_length: the predicted frames 1..16
        teacher_batch_size=32,
        target_adapter="avgpool",  # variant 1; "avgpool_conv" = variant 2; "strided_conv" = variant 3
        target_adapter_kernel_size=3,
        target_adapter_depthwise=True,
        target_grid_thw=(4, 5, 5),  # per-view MoT token grid: 4 predicted latent frames x 5 x 5 (160-px content)
        target_subgrid_thw=(1, 1, 1),  # teacher cells per MoT token; (1, 2, 2) = less pooling (hs11 v5 / hs10 v13-v14)
        student_upsampler="none",  # "trilinear" = upsample the student to the full teacher grid (hs11 v12)
        projector_type="mlp",  # REPA MLP (4096 -> 2048 -> 2048 -> D_t); "linear" = single Linear(4096, D_t)
        projector_hidden_dim=2048,
        num_views=2,  # third-person | wrist, concatenated along width
        native_video_key="video_native",
        center_targets=False,
    )
    # Optional SIGReg regularizer on the REPA projector output (off; enable from TOML like the Edge v9 recipe).
    cfg["sigreg"] = dict(
        enabled=False,
        input="repa_projection",
        normalize_by_count=True,
        loss_weight=1.0,
        layer_index=10,
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
    """LIBERO-10 PackingDataLoader; train and val share everything but the episode subset / dropout / seed.

    Same wiring as ``action_policy_libero_edge._libero_packing_loader`` so the TOML knobs
    ``[dataloader_{train,val}].episode_subset_path`` / ``max_samples_per_batch`` land on the
    same Hydra paths for both tiers."""
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
            # Shuffling is handled by the dataset (iterable_shuffle=True below):
            # ActionIterableShuffleDataset streams rank x worker-sharded, episode-order-
            # shuffled, sequential-within-episode.
            datasets=dict(
                libero=dict(
                    ratio=1,
                    dataset=L(get_action_libero_sft_dataset)(
                        # Local LeRobot dir for the libero_10 suite ONLY. Use the
                        # 20 FPS nvidia/LIBERO_LeRobot_v3 (matches the bundled stats + 20 Hz eval):
                        #   hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset \
                        #     --include 'libero_10/**' --local-dir <dir>   # LIBERO_ROOT=<dir>/libero_10
                        root="${oc.env:LIBERO_ROOT}",
                        fps=20,  # metadata only (FPS-agnostic loader reads native fps from info.json)
                        chunk_length=16,
                        image_size=256,  # concat_view -> 256x512
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
                        format_prompt_as_json=True,  # structured JSON prompts (set False for plain-text)
                        tokenizer_config="${model.config.vlm_config.tokenizer}",
                        keep_native_video=True,  # ships uint8 [3,17,256,512] as data_batch["video_native"] (teacher input)
                    ),
                ),
            ),
        ),
    )


action_policy_libero_nano_repa = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            # FusedAdam with fp32 master_weights + eps 1e-8 (bf16 params + eps 1e-6
            # diverged on the action loss).
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
            name="action_policy_libero_nano_repa",
            wandb_mode="disabled",
        ),
        model=dict(
            config=_action_policy_libero_nano_repa_model_config(),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,  # popped by build_optimizer for FusedAdam (fused by construction)
            # Train the generation + action heads.
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "repa_",  # REPA projector (+ learnable target adapter for variants 2/3)
            ],
            lr=5.0e-05,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],  # smoke: 100 iters (real run sets via TOML, GA=10000)
            f_max=[1.0],
            f_min=[0.0],
            f_start=[1.0e-06],
            verbosity_interval=0,
            warm_up_steps=[0],  # smoke (real run sets via TOML, GA=2000)
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,  # real run sets via TOML (GA=2)
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
                # val/loss_total, val/flow_matching_loss_{action,vision}, val/repa_* (when run_validation)
                val_loss_breakdown=L(ValLossBreakdownCallback)(
                    keys=[
                        "flow_matching_loss_action",
                        "flow_matching_loss_vision",
                        "repa_loss",
                        "repa_weight",
                        "repa_weighted_loss",
                        "repa_sigma_frac",
                        "repa_cos_sim",
                        "repa_rel_loss",
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
            # Skip net_ema (EMA warm-starts from net, see dcp.py) and the action
            # heads, so they init fresh from the base (the public Cosmos3-Nano base
            # has no LIBERO-trained action heads).
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
                "repa_",  # REPA head is absent from the base checkpoint
            ],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Nano DCP dir; supply via TOML/env
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
            # Fixed few-shot subset (None = all 379 libero_10 episodes minus the 1% val split). Set from the
            # TOML via [dataloader_train].episode_subset_path, e.g. "libero_10_3ep_per_task_seed42.json".
            episode_subset_path=None,
            cfg_dropout_rate=0.1,
            episode_shuffle_seed=42,
            max_samples_per_batch=128,  # hs11 Nano recipes: 128 windows/rank (TOML-overridable)
            num_workers=4,
            prefetch_factor=4,
        ),
        # Held-out validation: 50 demos disjoint from the 3-demo training subset, no prompt dropout, its own
        # shuffle seed; every pass restarts at epoch 0 so successive val numbers use the same windows.
        # Enabled from the TOML ([trainer].run_validation / validation_iter / max_val_iter /
        # run_validation_on_start, [dataloader_val].episode_subset_path); off by default (run_validation=False).
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


for _item in [action_policy_libero_nano_repa]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
