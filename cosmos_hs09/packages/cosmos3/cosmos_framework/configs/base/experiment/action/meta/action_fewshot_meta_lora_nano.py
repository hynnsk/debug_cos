# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_fewshot_meta_lora_nano`` -- Cosmos3-NANO twin of ``action_fewshot_meta_lora_edge`` (cosmos_hs09, 2026-09-30).

Cross-embodiment first-order MAML in the LoRA regime on Cosmos3-Nano (Qwen3-VL-8B MoT: hidden 4096, 36 decoder blocks,
15.75 B parameters). theta_meta = action heads + action_modality_embed + LoRA(rank 32, alpha 64) on every ``*_moe_gen``
q/k/v/o AND the Qwen3 SwiGLU ``mlp_moe_gen.gate_proj / up_proj / down_proj`` + ``time_embedder`` -- the trainable set a
downstream Nano LoRA post-training must use (same rank / alpha / targets, ``lora_freeze_base=False`` there). Driven by
``cosmos_framework/scripts/train_action_meta.py`` exactly like the Edge experiment; the only model-side differences are
the Nano tier deltas of the LIBERO Nano recipe:

* ``NANO_MODEL_CONFIG`` + ``rectified_flow_training_config.loss_scale=10`` / ``image_loss_scale=None`` (the Nano default
  is 1.0; only matters for ``loss_mode="total"``), ``diffusion_expert_config.load_weights_from_pretrained=False`` (the
  DCP is the init), ``tokenizer.encode_exact_durations=[17, 61, 73]``, ``max_num_tokens_after_packing=45056``;
* ``lora_target_modules`` includes ``mlp_moe_gen.gate_proj`` (Qwen3 SwiGLU; Nemotron has no gate);
* ``checkpoint.keys_to_skip_loading`` also lists ``action_pos_embed`` (a key of the Nano DCP that the current network
  does not have, skipped by every Nano recipe).

Replicated bf16 model without FSDP (``parallelism.enable_inference_mode=True``, shard 1): 15.75 B x 2 B = 31.5 GB of
frozen weights per rank + theta_meta (~0.1 B params: fp32 fast weights, inner-Adam and outer-AdamW state ~2.5 GB) +
activations of one packed batch under full activation checkpointing. Sized for 8 x H200 (141 GB); a 1-GPU 48 GB card
only fits the smoke recipe (a few windows). Every rank runs its own meta-episode per iteration and only the theta_meta
gradient is all-reduced, so 8 ranks = 8 episodes per outer step.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.meta.action_fewshot_meta_lora_edge import META_EMBODIMENTS
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.action.meta.episodic_sampler import build_meta_episode_loader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

LORA_TARGET_MODULES_NANO = (
    "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen,"
    "mlp_moe_gen.gate_proj,mlp_moe_gen.up_proj,mlp_moe_gen.down_proj"
)


def _action_fewshot_meta_lora_nano_model_config() -> dict:
    cfg = copy.deepcopy(NANO_MODEL_CONFIG)
    cfg["action_gen"] = True
    cfg["vision_gen"] = True
    cfg["max_action_dim"] = 64
    cfg["max_num_tokens_after_packing"] = 45056
    # Nano few-shot recipe deltas (action_policy_libero_nano): the DCP is the init, x10 flow-matching scales.
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["rectified_flow_training_config"]["loss_scale"] = 10.0
    cfg["rectified_flow_training_config"]["image_loss_scale"] = None
    cfg["tokenizer"]["encode_exact_durations"] = [17, 61, 73]  # match Cosmos3 base + reference SFT (do NOT reduce)
    # Frozen backbone: no EMA, eager mode (the inner loop swaps weights in place), full activation checkpointing.
    cfg["ema"]["enabled"] = False
    cfg["compile"]["enabled"] = False
    cfg["activation_checkpointing"]["mode"] = "full"
    # No FSDP: replicated bf16 parameters so the adapter can write fast weights in place (see module docstring).
    cfg["parallelism"]["enable_inference_mode"] = True
    cfg["parallelism"]["data_parallel_shard_degree"] = 1
    cfg["parallelism"]["data_parallel_replicate_degree"] = 1
    cfg["parallelism"]["fsdp_master_dtype"] = "bfloat16"
    # LoRA adapters on the generation tower become part of theta_meta. Same rank / alpha / targets as the downstream
    # Nano LoRA post-training must use -- the meta init maps 1:1 onto the parameters the post-training optimizes.
    cfg["lora_enabled"] = True
    cfg["lora_rank"] = 32
    cfg["lora_alpha"] = 64
    cfg["lora_target_modules"] = LORA_TARGET_MODULES_NANO
    # MetaActionAdapter re-enables requires_grad on its own parameter set; the injector's freeze is fine here.
    cfg["lora_freeze_base"] = True
    return cfg


action_fewshot_meta_lora_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "adamw"},
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
            group="fewshot_meta",
            name="action_fewshot_meta_lora_nano",
            wandb_mode="disabled",
        ),
        model=dict(
            config=_action_fewshot_meta_lora_nano_model_config(),
        ),
        # The meta trainer builds its own outer optimizer from [custom.meta]; these entries only keep
        # the Hydra tree valid and document which parameters are meta-learned.
        optimizer=dict(
            lr=1.0e-04,
            weight_decay=0.0,
            keys_to_select=["action2llm", "llm2action", "action_modality_embed", "lora_", "time_embedder"],
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[2000],
            f_max=[1.0],
            f_min=[0.1],
            f_start=[1.0e-06],
            verbosity_interval=0,
            warm_up_steps=[50],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=10,
            max_iter=1000,
            run_validation=False,
            seed=42,
            timeout_period=999999999,
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # EMA is disabled; LoRA tensors are not in the base DCP (fresh A / B=0); action_pos_embed is a key of the
            # Nano DCP the current network does not have (skipped by every Nano recipe). Mid-trained action rows ARE loaded.
            keys_to_skip_loading=["net_ema.", "lora_", "action_pos_embed"],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Nano DCP dir; supply via TOML/env
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,  # theta_meta snapshot interval (meta_action_init_iter_XXXXXX.pt)
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
        dataloader_train=L(build_meta_episode_loader)(
            # Parent directory holding google_robot_rt1/, bridge_v2/, robomind/{ur,franka}_1rgb/, yam/repos/.
            data_root="${oc.env:ROBOT_FEWSHOT_ROOT}",
            embodiments=META_EMBODIMENTS,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            max_action_dim="${model.config.max_action_dim}",
            chunk_length=16,
            mode="wam",
            resolution="256",  # every embodiment is snapped onto the 256 tier canvases (LIBERO trains at 192x320)
            cfg_dropout_rate=0.1,
            format_prompt_as_json=True,
            append_idle_frames=True,
            dataset_kwargs=None,
            root_overrides=None,
            # ---- episodic sampler (overridable from [custom.meta]); the H200 recipe defaults ----
            k_shot=8,
            q_query=4,
            windows_per_demo=8,
            query_windows_per_demo=None,
            disjoint_tasks="auto",
            min_tasks_for_disjoint=4,
            min_windows_per_demo=1,
            embodiment_weights=None,
            max_samples_per_batch=64,
            num_workers=4,
            prefetch_factor=2,
            seed=42,
            embodiment_override=None,
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


for _item in [action_fewshot_meta_lora_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
