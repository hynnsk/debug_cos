"""cosmos_hs11 training configs for openpi (registered from ``openpi.training.config``).

LIBERO (NVIDIA LIBERO_LeRobot_v3, LeRobot v3.0) post-training recipes, one per (suite, data setting, init)::

    pi05_libero_v3_<suite>_<data>_<init>   suite in {libero_10, libero_goal, libero_object, libero_spatial}
                                           data  in {all (every episode), 3ep (3 demos/task), 1ep (1 demo/task)}
                                           init  in {base (pi05_base), reptile ($HS11_REPTILE_PARAMS)}

All recipes share the ``pi05_libero`` model (pi05, action horizon 10, continuous-state off), AdamW (clip 1.0),
EMA 0.999 and quantile normalization; the Reptile variant only swaps the initial weights. Norm stats are computed once
per suite from the ``*_all_base`` config and shared by the other five recipes of that suite (``AssetsConfig``).
"""

from __future__ import annotations

import dataclasses
import functools
import os

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from openpi.hs11 import libero_v3

PI05_BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"
# The Reptile recipes read the checkpoint to warm-start from here (``<reptile ckpt>/<step>/params``).
ENV_REPTILE_PARAMS = "HS11_REPTILE_PARAMS"
REPTILE_PARAMS_UNSET = "HS11_REPTILE_PARAMS_NOT_SET"

DATA_SETTINGS: dict[str, dict[str, str] | None] = {
    "all": None,
    "3ep": libero_v3.FINAL_3EP_SUBSETS,
    "1ep": libero_v3.FINAL_1EP_SUBSETS,
}
# Train steps per data setting (global batch 64): all ~37 epochs of 52 k frames; 3ep ~94 epochs of 3.4 k; 1ep ~170 of 1.1 k.
NUM_TRAIN_STEPS = {"all": 30_000, "3ep": 5_000, "1ep": 3_000}
SAVE_INTERVAL = {"all": 5_000, "3ep": 1_000, "1ep": 1_000}
INITS = ("base", "reptile")


def pi05_model() -> pi0_config.Pi0Config:
    """The ``pi05_libero`` model config: pi05, 10-step chunks, state NOT fed as discrete tokens (openpi's LIBERO recipe)."""
    return pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)


def reptile_params_path() -> str:
    return os.environ.get(ENV_REPTILE_PARAMS, REPTILE_PARAMS_UNSET)


def libero_config_name(suite: str, data: str, init: str) -> str:
    return f"pi05_libero_v3_{suite}_{data}_{init}"


def norm_stats_owner(suite: str) -> str:
    """The config whose ``scripts/compute_norm_stats.py`` run produces the suite's statistics."""
    return libero_config_name(suite, "all", "base")


def get_hs11_configs():
    # Imported here to avoid a circular import (openpi.training.config imports this module while initializing).
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import DataConfigFactory
    from openpi.training.config import ModelTransformFactory
    from openpi.training.config import TrainConfig

    @dataclasses.dataclass(frozen=True)
    class LiberoV3DataConfig(DataConfigFactory):
        """LIBERO_LeRobot_v3 suite, optionally restricted to a fixed episode subset json."""

        suite: str = "libero_goal"
        episode_subset: str | None = None
        # Training-time prompt dropout probability (0 = off); see libero_v3.PromptDropout.
        prompt_dropout: float = 0.0

        def create(self, assets_dirs, model_config) -> DataConfig:
            spec = libero_v3.make_libero_v3_spec(self.suite, self.episode_subset)
            repack = [_transforms.RepackTransform(dict(libero_v3.REPACK))]
            if self.prompt_dropout > 0.0:
                repack.append(libero_v3.PromptDropout(p=float(self.prompt_dropout)))
            ranking = getattr(model_config, "language_ranking", None)
            if ranking is not None and ranking.enabled:
                # Language ranking: every sample also carries a wrong-task instruction (after the dropout, so a dropped
                # prompt is flagged invalid). Tokenized by TokenizeNegativePrompt in the model transforms.
                repack.append(libero_v3.NegativePrompt(instructions=libero_v3.suite_instructions(spec)))
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=_transforms.Group(inputs=repack),
                data_transforms=_transforms.Group(
                    inputs=[libero_v3.LiberoV3Inputs(model_type=model_config.model_type)],
                    outputs=[libero_v3.LiberoV3Outputs()],
                ),
                model_transforms=ModelTransformFactory()(model_config),
                prompt_from_task=False,
                dataset_builder=functools.partial(libero_v3.build_libero_v3_dataset, spec),
            )

    configs = []
    for suite in libero_v3.LIBERO_SUITES:
        owner = norm_stats_owner(suite)
        for data, subsets in DATA_SETTINGS.items():
            subset = None if subsets is None else subsets[suite]
            steps = NUM_TRAIN_STEPS[data]
            if data == "all":
                # pi05_libero's schedule: 10 k warmup then constant 5e-5.
                lr_schedule = _optimizer.CosineDecaySchedule(warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5)
            else:
                lr_schedule = _optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=5e-5, decay_steps=steps, decay_lr=5e-6)
            for init in INITS:
                name = libero_config_name(suite, data, init)
                params_path = PI05_BASE_PARAMS if init == "base" else reptile_params_path()
                configs.append(
                    TrainConfig(
                        name=name,
                        project_name="openpi_hs11",
                        model=pi05_model(),
                        data=LiberoV3DataConfig(
                            repo_id=suite,
                            suite=suite,
                            episode_subset=subset,
                            assets=AssetsConfig() if name == owner else AssetsConfig(assets_dir=f"./assets/{owner}"),
                            base_config=DataConfig(prompt_from_task=False),
                        ),
                        weight_loader=weight_loaders.CheckpointWeightLoader(params_path),
                        batch_size=64,
                        num_workers=8,
                        lr_schedule=lr_schedule,
                        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
                        ema_decay=0.999,
                        num_train_steps=steps,
                        save_interval=SAVE_INTERVAL[data],
                        keep_period=SAVE_INTERVAL[data],
                        log_interval=50,
                        fsdp_devices=1,
                        wandb_enabled=False,
                        policy_metadata={
                            "suite": suite,
                            "data": data,
                            "init": init,
                            "action_convention": "libero_v3",
                            "episode_subset": subset,
                        },
                    )
                )
    return configs
