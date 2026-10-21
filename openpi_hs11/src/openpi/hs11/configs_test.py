import json
import os
import pathlib

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import openpi.training.config as _config
from openpi.hs11 import configs as hs11_configs
from openpi.hs11 import libero_v3

ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_all_recipes_are_registered_with_the_pi05_libero_model():
    names = [n for n in _config._CONFIGS_DICT if n.startswith("pi05_libero_v3_")]  # noqa: SLF001
    assert len(names) == 4 * 3 * 2
    for suite in libero_v3.LIBERO_SUITES:
        for data in hs11_configs.DATA_SETTINGS:
            for init in hs11_configs.INITS:
                cfg = _config.get_config(hs11_configs.libero_config_name(suite, data, init))
                assert cfg.model.pi05 and cfg.model.action_horizon == 10 and cfg.model.discrete_state_input is False
                assert cfg.num_train_steps == hs11_configs.NUM_TRAIN_STEPS[data]
                assert cfg.ema_decay == 0.999 and cfg.optimizer.clip_gradient_norm == 1.0
                expected = hs11_configs.PI05_BASE_PARAMS if init == "base" else hs11_configs.reptile_params_path()
                assert cfg.weight_loader.params_path == expected
                assert cfg.data.suite == suite and cfg.data.repo_id == suite
                subset = None if data == "all" else hs11_configs.DATA_SETTINGS[data][suite]
                assert cfg.data.episode_subset == subset
                if subset is not None:
                    assert (ROOT / "assets" / "episode_subsets" / subset).is_file()
                owner = hs11_configs.norm_stats_owner(suite)
                if cfg.name == owner:
                    assert cfg.data.assets.assets_dir is None
                else:
                    assert cfg.data.assets.assets_dir == f"./assets/{owner}"


def test_subset_jsons_are_one_or_three_demos_per_task():
    for data, subsets in hs11_configs.DATA_SETTINGS.items():
        if subsets is None:
            continue
        for suite, name in subsets.items():
            spec = json.loads((ROOT / "assets" / "episode_subsets" / name).read_text())
            per_task = [len(v["episodes"] if isinstance(v, dict) else v) for v in spec["tasks"].values()]
            assert len(per_task) == 10 and set(per_task) == {3 if data == "3ep" else 1}, (suite, name)
            assert spec["source_root_basename"] == suite


def test_data_config_builds_transforms_without_norm_stats(tmp_path):
    # the *_all_base owner reads stats from assets_dirs (tmp_path here -> none); the other recipes point at the owner's dir
    cfg = _config.get_config("pi05_libero_v3_libero_goal_all_base")
    dc = cfg.data.create(tmp_path, cfg.model)  # no stats under tmp_path -> None, no exception
    assert dc.norm_stats is None and dc.use_quantile_norm and dc.dataset_builder is not None
    assert [type(t).__name__ for t in dc.data_transforms.inputs] == ["LiberoV3Inputs"]
    assert dc.repack_transforms.inputs[0].structure == dict(libero_v3.REPACK)


def test_v3_action_to_env_gripper_mapping():
    import numpy as np

    a = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], [0.0] * 6 + [0.0]], dtype=np.float32)
    env = libero_v3.v3_action_to_env(a)
    assert env[0, 6] == -1.0 and env[1, 6] == 1.0 and np.allclose(env[:, :6], a[:, :6])
    assert libero_v3.gripper_v3_to_env(0.9, binarize=False) == pytest.approx(-0.8)
    assert libero_v3.gripper_v3_to_env(0.49) == 1.0 and libero_v3.gripper_v3_to_env(0.51) == -1.0
