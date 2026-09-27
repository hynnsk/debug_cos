# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_reptile_meta_nano`` -- Reptile meta-training of Cosmos3-Nano (Qwen3-VL-8B MoT) over robot embodiments.

Same algorithm, trainer (``cosmos_framework/scripts/train_action_reptile.py``) and ``[custom.meta]`` knobs as
``action_reptile_meta_edge`` (cosmos_hs11); only the model tier differs:

* ``NANO_MODEL_CONFIG``: hidden 4096, 36 MoT blocks, 15.75B parameters of which the generation tower is 6.95B
  (vs Edge 3.37B / 1.41B). The few-shot LIBERO Nano recipe's model deltas are mirrored so theta lands in the
  downstream model: ``rectified_flow_training_config.loss_scale=10`` / ``image_loss_scale=None`` (the Nano default
  is 1.0 -- with it the inner objective would not be the 10 x vision + 10 x action of post-training),
  ``diffusion_expert_config.load_weights_from_pretrained=False`` (the DCP is the init), ``tokenizer.encode_exact_durations``
  [17, 61, 73], ``max_num_tokens_after_packing`` 45056.
* ``compile.enabled=False``: mandatory on 48 GB Ampere (Inductor's fused RMSNorm backward of the 4096-wide language
  region exceeds the 100 KB shared memory; see the cosmos_hs08 Nano notes) and required by Reptile anyway.
* FSDP shard 8 = one 8 x 48 GB node. Persistent per-GPU memory in full mode: fp32 master weights 63 GB + grads 28 GB
  + Adam 56 GB + theta copy 28 GB, all / 8 = ~22 GB, plus ~0.22 GB per 256-res window of activations (full activation
  checkpointing; measured on the hs08 Nano post-training). 32 windows/rank -> ~30 GB; 64 -> ~37 GB (A6000 only);
  128 -> ~50 GB (does not fit any 48 GB card).
* theta (full mode) = the trainable set of ``action_policy_libero_nano``: moe_gen + time_embedder + vae2llm +
  llm2vae + action heads (Nano has no ``k_norm_und_for_gen``). LoRA mode targets add ``mlp_moe_gen.gate_proj``
  (Qwen3 SwiGLU MLP).
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.meta.action_reptile_meta_edge import META_EMBODIMENTS
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.action.meta.episodic_sampler import build_reptile_episode_loader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

# theta (= trainable set) of the two modes. FULL == optimizer.keys_to_select of action_policy_libero_nano.
FULL_FT_TRAINABLE_KEYS_NANO = [
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "llm2action",
    "action_modality_embed",
]
LORA_TRAINABLE_KEYS_NANO = ["lora_", "action2llm", "llm2action", "action_modality_embed", "time_embedder"]
LORA_TARGET_MODULES_NANO = (
    "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen,"
    "mlp_moe_gen.gate_proj,mlp_moe_gen.up_proj,mlp_moe_gen.down_proj"
)


def _action_reptile_meta_nano_model_config() -> dict:
    cfg = copy.deepcopy(NANO_MODEL_CONFIG)  # action_gen=True, max_action_dim=64
    cfg["action_gen"] = True
    cfg["vision_gen"] = True
    cfg["max_action_dim"] = 64
    cfg["max_num_tokens_after_packing"] = 45056  # few-shot value (the full-data Nano recipe uses 74000)
    # few-shot LIBERO Nano recipe deltas (action_policy_libero_nano)
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["rectified_flow_training_config"]["loss_scale"] = 10.0
    cfg["rectified_flow_training_config"]["image_loss_scale"] = None
    cfg["tokenizer"]["encode_exact_durations"] = [17, 61, 73]  # match Cosmos3 base + reference SFT (do NOT reduce)
    # Reptile: theta is the smoothed quantity (no EMA), weights are rewritten every meta step (no compile;
    # compile is also unusable on 48 GB Ampere for the 4096-wide Nano), full activation checkpointing.
    cfg["ema"]["enabled"] = False
    cfg["compile"]["enabled"] = False
    cfg["activation_checkpointing"]["mode"] = "full"
    cfg["parallelism"]["enable_inference_mode"] = False
    cfg["parallelism"]["data_parallel_shard_degree"] = 8
    cfg["parallelism"]["data_parallel_replicate_degree"] = 1
    cfg["parallelism"]["fsdp_master_dtype"] = "float32"
    cfg["lora_enabled"] = False
    cfg["lora_rank"] = 32
    cfg["lora_alpha"] = 64
    cfg["lora_target_modules"] = LORA_TARGET_MODULES_NANO
    cfg["lora_freeze_base"] = False  # LoRA + heads + time_embedder train together; keys_to_select selects
    return cfg


action_reptile_meta_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /checkpoint": "s3"},
            {"override /callbacks": ["basic"]},
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3_action_meta",
            group="reptile_meta",
            name="action_reptile_meta_nano",
            wandb_mode="disabled",
        ),
        model=dict(
            config=_action_reptile_meta_nano_model_config(),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            keys_to_select=list(FULL_FT_TRAINABLE_KEYS_NANO),
            lr=5.0e-05,  # = the Nano few-shot post-training LR (action_policy_libero_nano)
            lr_multipliers={
                "action2llm": 1.0,
                "llm2action": 1.0,
                "action_modality_embed": 1.0,
                "lora_": 1.0,  # present so a TOML can override it (Hydra cannot add dict keys)
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[10000000],
            f_max=[1.0],
            f_min=[1.0],
            f_start=[1.0],
            verbosity_interval=0,
            warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=10,
            max_iter=1000,  # meta-iterations (each = inner_steps optimizer steps + 1 meta step)
            run_validation=False,
            seed=42,
            timeout_period=999999999,
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # The released Cosmos3-Nano ships DROID-trained action heads; like the LIBERO recipes they start fresh
            # (scratch row 31 is meta-learned). action_pos_embed is skipped for parity with action_policy_libero_nano.
            keys_to_skip_loading=["net_ema.", "action2llm", "llm2action", "action_modality_embed", "action_pos_embed"],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Nano DCP dir; supply via TOML/env
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=500,  # a full-mode Nano DCP is ~120 GB (63 GB fp32 weights + Adam moments); 2 snapshots per run
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
        dataloader_train=L(build_reptile_episode_loader)(
            data_root="${oc.env:ROBOT_FEWSHOT_ROOT}",
            embodiments=META_EMBODIMENTS,
            tokenizer_config="${model.config.vlm_config.tokenizer}",  # Nano's Qwen text tokenizer
            max_action_dim="${model.config.max_action_dim}",
            chunk_length=16,
            mode="wam",
            resolution="256",
            cfg_dropout_rate=0.1,
            format_prompt_as_json=True,
            append_idle_frames=True,
            dataset_kwargs=None,
            root_overrides=None,
            k_shot=16,
            q_query=8,
            windows_per_demo=16,
            query_windows_per_demo=None,
            disjoint_tasks=False,
            min_tasks_for_disjoint=4,
            min_windows_per_demo=1,
            embodiment_weights=None,
            max_samples_per_batch=32,  # per rank; 8 ranks x 32 = 256 = one pass over the 16 x 16 support set
            num_workers=8,
            prefetch_factor=2,
            loader_timeout_s=1800.0,
            seed=42,
            embodiment_override=None,
            spec_offset=0,
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


for _item in [action_reptile_meta_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
