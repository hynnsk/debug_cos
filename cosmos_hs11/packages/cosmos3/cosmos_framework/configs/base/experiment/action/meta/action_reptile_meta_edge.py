# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_reptile_meta_edge`` -- Reptile meta-training of Cosmos3-Edge over robot embodiments (cosmos_hs11).

Sits between Cosmos3 mid-training and the LIBERO few-shot post-training::

    Cosmos3-Edge (mid-trained)
        ---- Reptile over {RT-1, Bridge V2, RoboMIND-UR, RoboMIND-Franka, MolmoAct2-YAM} ---->
    theta (DCP checkpoint + meta_action_init.pt)  --->  LIBERO post-training warm-started from theta

Driven by ``cosmos_framework/scripts/train_action_reptile.py``. Unlike the FOMAML recipes (hs07/hs09) the model
is trained with the ORDINARY stack: FSDP2 sharding (``data_parallel_shard_degree=4``), fp32 master weights,
FusedAdam from ``[optimizer]`` (``keys_to_select`` = the trainable set = theta), activation checkpointing. The
inner loop is k steps of that optimizer on one embodiment's K demonstrations; the meta step interpolates
theta toward the result. Defaults are the FULL-parameter mode (theta = the trainable set of the Cosmos
full-FT post-training recipe ``action_policy_libero_edge``); the LoRA mode is a TOML change
(``examples/toml/sft_config/action_reptile_meta_lora_edge.toml``).

Model-side choices: ``ema.enabled=False`` (theta itself is the smoothed quantity), ``compile.enabled=False``
(weights are rewritten every meta step), ``activation_checkpointing.mode="full"``. Action heads are skipped
at load and start from the Cosmos fresh init in the scratch row (like the LIBERO baseline's fresh heads).
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.meta.episodic_sampler import build_reptile_episode_loader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

META_EMBODIMENTS = ["fractal", "bridge", "robomind_ur", "robomind_franka", "molmoact2_yam"]

# theta (= trainable set) of the two modes. FULL == optimizer.keys_to_select of action_policy_libero_edge.
FULL_FT_TRAINABLE_KEYS = [
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "k_norm_und_for_gen",
    "action2llm",
    "llm2action",
    "action_modality_embed",
]
LORA_TRAINABLE_KEYS = ["lora_", "action2llm", "llm2action", "action_modality_embed", "time_embedder"]
LORA_TARGET_MODULES = (
    "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen,mlp_moe_gen.up_proj,mlp_moe_gen.down_proj"
)


def _action_reptile_meta_edge_model_config() -> dict:
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
    cfg["action_gen"] = True
    cfg["vision_gen"] = True
    cfg["max_action_dim"] = 64
    cfg["max_num_tokens_after_packing"] = 45056
    cfg["ema"]["enabled"] = False
    cfg["compile"]["enabled"] = False
    cfg["activation_checkpointing"]["mode"] = "full"
    # Ordinary FSDP2 training of the trainable set: 4-way sharding, fp32 master weights so that k small
    # Adam steps (lr 5e-5) are not rounded away in bf16.
    cfg["parallelism"]["enable_inference_mode"] = False
    cfg["parallelism"]["data_parallel_shard_degree"] = 4
    cfg["parallelism"]["data_parallel_replicate_degree"] = 1
    cfg["parallelism"]["fsdp_master_dtype"] = "float32"
    # LoRA mode is opt-in from the TOML ([model] lora_enabled=true + LORA keys_to_select). Same rank/alpha/
    # targets as the downstream LoRA recipe so theta maps 1:1.
    cfg["lora_enabled"] = False
    cfg["lora_rank"] = 32
    cfg["lora_alpha"] = 64
    cfg["lora_target_modules"] = LORA_TARGET_MODULES
    cfg["lora_freeze_base"] = False  # LoRA + heads + time_embedder train together; keys_to_select selects
    return cfg


action_reptile_meta_edge = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            # FusedAdam with fp32 master_weights + eps 1e-8, as in the LIBERO post-training recipe.
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
            name="action_reptile_meta_edge",
            wandb_mode="disabled",
        ),
        model=dict(
            config=_action_reptile_meta_edge_model_config(),
        ),
        # The INNER optimizer (k steps per meta-episode). keys_to_select == theta. Mirrors the Cosmos full-FT
        # recipe except the action-head multiplier (1x: the heads are meta-learned across episodes, not fresh
        # per episode) -- the same 1x the downstream metainit recipes use.
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            keys_to_select=list(FULL_FT_TRAINABLE_KEYS),
            lr=5.0e-05,
            lr_multipliers={
                "action2llm": 1.0,
                "llm2action": 1.0,
                "action_modality_embed": 1.0,
                "lora_": 1.0,  # present so a TOML can override it (Hydra cannot add dict keys)
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        # Never stepped by the Reptile trainer (the inner LR schedule is set directly: linear ramp over
        # inner_warmup_steps, then constant). Present so the standard checkpointer can save/load it.
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
            # No EMA. Action heads start fresh (Cosmos init policy, scratch row) exactly like the LIBERO
            # baseline's heads; everything else warm-starts from the mid-trained base.
            keys_to_skip_loading=["net_ema.", "action2llm", "llm2action", "action_modality_embed"],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Edge DCP dir; supply via TOML/env
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=250,  # DCP snapshot of theta (~24 GB each in full mode) + meta_action_init_iter_XXXXXX.pt
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
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            max_action_dim="${model.config.max_action_dim}",
            chunk_length=16,
            mode="wam",
            resolution="256",
            cfg_dropout_rate=0.1,
            format_prompt_as_json=True,
            append_idle_frames=True,
            dataset_kwargs=None,
            root_overrides=None,
            # ---- episodic sampler (overridable from [custom.meta]) ----
            # K=8 demos x 16 windows = 128 support windows shared by the 4 ranks (32 each = one inner step
            # = one pass over the support set); Q=4 x 16 = 64 query windows for the zero-shot/adapted diagnostics.
            k_shot=8,
            q_query=4,
            windows_per_demo=16,
            query_windows_per_demo=None,
            disjoint_tasks="auto",
            min_tasks_for_disjoint=4,
            min_windows_per_demo=1,
            embodiment_weights=None,
            max_samples_per_batch=32,  # per-rank windows per inner step
            num_workers=4,
            prefetch_factor=2,
            loader_timeout_s=1800.0,
            seed=42,
            embodiment_override=None,
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


for _item in [action_reptile_meta_edge]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
