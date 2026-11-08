# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_fewshot_meta_edge`` -- cross-embodiment few-shot meta-training of the Cosmos3-Edge action heads.

Sits between Cosmos3 mid-training and the LIBERO few-shot post-training (docs/action_fewshot_meta.md):

    Cosmos3-Edge (mid-trained, frozen)
        + theta_meta (action2llm / llm2action / action_modality_embed, one shared init)
        ---- first-order MAML over {RT-1, Bridge, RoboMIND-UR, RoboMIND-Franka, MolmoAct2-YAM} ---->
    meta_action_init.pt  --->  LIBERO post-training with checkpoint.meta_action_init_path

Driven by ``cosmos_framework/scripts/train_action_meta.py`` (NOT ``scripts/train.py``): the meta loop
needs support -> inner update -> query -> outer update, which the standard trainer cannot express.
Model-side choices that differ from the post-training recipe and why:

* ``parallelism.enable_inference_mode=True`` + ``data_parallel_shard_degree=1``: the 3.4B backbone is
  frozen, so every rank keeps a full bf16 replica with *plain* parameters (no FSDP2 DTensors). This
  is what lets the adapter write fast weights into one DomainAwareLinear row in place. Only the tiny
  theta_meta gradient is all-reduced across ranks.
* ``ema.enabled=False`` / ``compile.enabled=False``: no EMA of a frozen network; eager mode avoids
  recompiles while the inner loop swaps weights.
* ``checkpoint.keys_to_skip_loading=["net_ema."]`` only: the mid-trained action rows ARE loaded so
  ``[custom.meta].init_source = "checkpoint_row" / "checkpoint_mean"`` can start theta_meta from them
  (default ``"fresh"`` matches the LIBERO baseline's fresh init).

Sampler / adapter knobs live in ``[custom.meta]`` of the run TOML (see
``examples/toml/sft_config/action_fewshot_meta_edge.toml``).
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.meta.episodic_sampler import build_meta_episode_loader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

META_EMBODIMENTS = ["fractal", "bridge", "robomind_ur", "robomind_franka", "molmoact2_yam"]


def _action_fewshot_meta_edge_model_config() -> dict:
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
    cfg["action_gen"] = True
    cfg["vision_gen"] = True
    cfg["max_action_dim"] = 64
    cfg["max_num_tokens_after_packing"] = 45056
    # Frozen backbone: no EMA, eager mode, full activation checkpointing (backward still traverses the
    # whole network to reach action2llm).
    cfg["ema"]["enabled"] = False
    cfg["compile"]["enabled"] = False
    cfg["activation_checkpointing"]["mode"] = "full"
    # No FSDP: replicated bf16 parameters (see module docstring).
    cfg["parallelism"]["enable_inference_mode"] = True
    cfg["parallelism"]["data_parallel_shard_degree"] = 1
    cfg["parallelism"]["fsdp_master_dtype"] = "bfloat16"
    return cfg


action_fewshot_meta_edge = LazyDict(
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
            name="action_fewshot_meta_edge",
            wandb_mode="disabled",
        ),
        model=dict(
            config=_action_fewshot_meta_edge_model_config(),
        ),
        # The meta trainer builds its own outer optimizer from [custom.meta]; these entries only keep
        # the Hydra tree valid and document which parameters are meta-learned.
        optimizer=dict(
            lr=1.0e-04,
            weight_decay=0.0,
            keys_to_select=["action2llm", "llm2action", "action_modality_embed"],
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
            max_iter=2000,
            run_validation=False,
            seed=42,
            timeout_period=999999999,
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # EMA is disabled; load everything else including the mid-trained action rows.
            keys_to_skip_loading=["net_ema."],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Edge DCP dir; supply via TOML/env
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
            # ---- episodic sampler (overridable from [custom.meta]) ----
            k_shot=5,
            q_query=5,
            windows_per_demo=8,
            query_windows_per_demo=None,
            disjoint_tasks="auto",
            min_tasks_for_disjoint=4,
            min_windows_per_demo=1,
            embodiment_weights=None,
            max_samples_per_batch=40,
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


for _item in [action_fewshot_meta_edge]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
