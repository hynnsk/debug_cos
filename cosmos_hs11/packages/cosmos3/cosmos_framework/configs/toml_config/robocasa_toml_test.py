# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""cosmos_hs11 RoboCasa365 Edge recipes: the six action_policy_robocasa_edge_{baseinit,reptileinit}_{10,30,100}pct TOMLs
validate, route to the experiment, differ only where they should (init / data fraction), the data fraction reaches the
nested robocasa dataset node, the experiment carries the RoboCasa contract on the LIBERO Edge trainable set, the reader's
per-task subsampling nests across fractions, and the launchers / eval script reference the right files and ports."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

ROOT = Path(__file__).resolve().parents[3]
TOML_DIR = ROOT / "examples/toml/sft_config"
INITS = ("baseinit", "reptileinit", "reptileinit_freshheads")
FRACTIONS = {10: 50, 30: 150, 100: None}
_NODE = "dataloader_train.dataloader.datasets.robocasa.dataset.episodes_per_task"


def _toml(init: str, frac: int) -> dict:
    return tomllib.load(open(TOML_DIR / f"action_policy_robocasa_edge_{init}_{frac}pct.toml", "rb"))


# Batch geometry is free per recipe (4 x 128 on 4 x A40, 2 x 256 on 2 x H200, ...) as long as the global batch stays
# 512 windows/step; wandb_mode is the user's call per run. Everything else must agree across fractions / inits.
_GEOMETRY = (
    ("model", "parallelism", "data_parallel_shard_degree"),
    ("dataloader_train", "max_samples_per_batch"),
    ("job", "wandb_mode"),
)
GLOBAL_BATCH = 512


def _global_batch(raw: dict) -> int:
    par = raw["model"]["parallelism"]
    assert par.get("data_parallel_replicate_degree", 1) == 1, (
        "pure FSDP: replicate must stay 1 (it is 'auto' = NPROC / shard)"
    )
    return (
        par["data_parallel_shard_degree"]
        * raw["dataloader_train"]["max_samples_per_batch"]
        * raw["trainer"].get("grad_accum_iter", 1)
    )


def _without_geometry(raw: dict) -> dict:
    import copy

    r = copy.deepcopy(raw)
    for *parents, leaf in _GEOMETRY:
        d = r
        for k in parents:
            d = d[k]
        d.pop(leaf, None)
    return r


@pytest.mark.parametrize("init", INITS)
@pytest.mark.parametrize("frac", sorted(FRACTIONS))
def test_toml_validates_and_routes_the_data_fraction(init: str, frac: int) -> None:
    raw = _toml(init, frac)
    cfg = SFTExperimentConfig.model_validate(raw)
    overrides = build_hydra_overrides(raw)
    assert "experiment=action_policy_robocasa_edge" in overrides
    assert cfg.job.name == f"edge_robocasa365_atomic_{frac}pct_fullft_{init}"
    n = FRACTIONS[frac]
    if n is None:
        assert "episodes_per_task" not in raw["dataloader_train"]
        assert not any(o.startswith(_NODE) for o in overrides)
    else:
        assert raw["dataloader_train"]["episodes_per_task"] == n
        assert f"{_NODE}={n}" in overrides
    # the same step budget for every fraction (as in RoboCasa365 Table 2), pure FSDP with global batch 512, validation on
    assert raw["trainer"]["max_iter"] == raw["scheduler"]["cycle_lengths"][0] == 3000
    assert raw["model"]["parallelism"]["data_parallel_shard_degree"] in (2, 4)
    assert _global_batch(raw) == GLOBAL_BATCH, (init, frac)
    assert raw["model"]["max_num_tokens_after_packing"] == 45056
    assert raw["trainer"]["run_validation"] is True and raw["dataloader_val"]["max_samples_per_batch"] == 16
    assert raw["model"]["lora_enabled"] is False


@pytest.mark.parametrize("frac", sorted(FRACTIONS))
def test_baseinit_and_reptileinit_differ_only_in_init_and_head_lr(frac: int) -> None:
    base, rep = _without_geometry(_toml("baseinit", frac)), _without_geometry(_toml("reptileinit", frac))
    for section in ("model", "scheduler", "trainer", "dataloader_train", "dataloader_val"):
        assert base[section] == rep[section], section
    assert {k: v for k, v in base["job"].items() if k != "name"} == {k: v for k, v in rep["job"].items() if k != "name"}
    assert base["optimizer"]["lr"] == rep["optimizer"]["lr"] == 5e-5
    assert base["optimizer"]["lr_multipliers"] == {"action2llm": 5.0, "llm2action": 5.0, "action_modality_embed": 5.0}
    assert rep["optimizer"]["lr_multipliers"] == {"action2llm": 1.0, "llm2action": 1.0, "action_modality_embed": 1.0}
    # init
    assert base["checkpoint"]["load_path"] == "${oc.env:BASE_CHECKPOINT_PATH}"
    assert "meta_action_init_path" not in base["checkpoint"] and "keys_to_skip_loading" not in base["checkpoint"]
    assert rep["checkpoint"]["load_path"] == "${oc.env:REPTILE_CKPT_PATH}"
    assert rep["checkpoint"]["meta_action_init_path"] == "${oc.env:META_ACTION_INIT_PATH}"
    assert rep["checkpoint"]["meta_action_init_include_lora"] is False
    assert rep["checkpoint"]["keys_to_skip_loading"] == [
        "net_ema.",
        "action2llm",
        "llm2action",
        "action_modality_embed",
    ]
    from cosmos_framework.data.generator.action.utils.domain_utils import EMBODIMENT_TO_DOMAIN_ID

    assert rep["checkpoint"]["meta_action_init_domain_id"] == EMBODIMENT_TO_DOMAIN_ID["robocasa"] == 30


@pytest.mark.parametrize("frac", sorted(FRACTIONS))
def test_freshheads_ablation_is_reptileinit_with_fresh_5x_heads(frac: int) -> None:
    rep, fresh = _without_geometry(_toml("reptileinit", frac)), _without_geometry(_toml("reptileinit_freshheads", frac))
    for section in ("model", "scheduler", "trainer", "dataloader_train", "dataloader_val"):
        assert rep[section] == fresh[section], section
    assert fresh["job"]["name"] == f"edge_robocasa365_atomic_{frac}pct_fullft_reptileinit_freshheads"
    assert (
        fresh["optimizer"]["lr_multipliers"] == _toml("baseinit", frac)["optimizer"]["lr_multipliers"]
    )  # 5x, as baseinit
    assert fresh["checkpoint"]["load_path"] == "${oc.env:REPTILE_CKPT_PATH}"  # the Reptile trunk ...
    assert not any(k.startswith("meta_action_init") for k in fresh["checkpoint"])  # ... without the meta heads
    assert fresh["checkpoint"]["keys_to_skip_loading"] == rep["checkpoint"]["keys_to_skip_loading"]


@pytest.mark.parametrize("init", INITS)
def test_fractions_differ_only_in_name_and_episodes_per_task(init: str) -> None:
    ref = _without_geometry(_toml(init, 100))
    for frac in (10, 30):
        raw = _without_geometry(_toml(init, frac))
        for section in raw:
            if section == "job":
                assert {k: v for k, v in raw[section].items() if k != "name"} == {
                    k: v for k, v in ref[section].items() if k != "name"
                }
            elif section == "dataloader_train":
                assert {k: v for k, v in raw[section].items() if k != "episodes_per_task"} == ref[section]
            else:
                assert raw[section] == ref[section], (init, frac, section)


def test_experiment_carries_the_robocasa_contract_on_the_libero_edge_trainable_set(monkeypatch) -> None:
    import cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_edge as libero
    import cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_robocasa_edge as m
    from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import DEFAULT_ALL_ATOMIC_TASKS

    monkeypatch.setenv(
        "ROBOCASA_ROOT", "/data/robocasa365/v30/target/atomic"
    )  # the LazyDict resolves ${oc.env:...} on read
    exp = m.action_policy_robocasa_edge
    assert exp["optimizer"]["keys_to_select"] == libero.action_policy_libero_edge["optimizer"]["keys_to_select"]
    assert "k_norm_und_for_gen" in exp["optimizer"]["keys_to_select"]
    assert exp["model"]["config"] == libero.action_policy_libero_edge["model"]["config"]
    assert (
        exp["checkpoint"]["keys_to_skip_loading"]
        == libero.action_policy_libero_edge["checkpoint"]["keys_to_skip_loading"]
    )
    for loader, split, n_workers in (("dataloader_train", "train", 4), ("dataloader_val", "val", 2)):
        node = exp[loader]["dataloader"]["datasets"]["robocasa"]["dataset"]
        assert node["root"] == "/data/robocasa365/v30/target/atomic"
        assert tuple(node["task_names"]) == DEFAULT_ALL_ATOMIC_TASKS and len(DEFAULT_ALL_ATOMIC_TASKS) == 18
        assert node["use_base_action"] is True and node["base_encoding"] == "raw_armfirst"  # arm block first
        assert node["camera_set"] == "left_wrist" and node["use_state"] is True
        assert node["chunk_length"] == 16 and node["fps"] == 20  # LIBERO / Reptile window contract, not Nano's 32
        assert node["action_normalization"] is None and node["format_prompt_as_json"] is True
        assert node["split"] == split and node["split_val_ratio"] == 0.01 and node["split_seed"] == 42
        assert node["episodes_per_task"] is None and node["episode_subset_seed"] == 42
        assert node["iterable_shuffle"] is True
        assert exp[loader]["dataloader"]["num_workers"] == n_workers
    assert exp["dataloader_train"]["dataloader"]["datasets"]["robocasa"]["dataset"]["cfg_dropout_rate"] == 0.1
    assert exp["dataloader_val"]["dataloader"]["datasets"]["robocasa"]["dataset"]["cfg_dropout_rate"] == 0.0
    assert "val_loss_breakdown" in exp["trainer"]["callbacks"]


def test_armfirst_contract_puts_the_10d_arm_block_first_everywhere() -> None:
    """Reader layout, state-token padding, action spec and the eval decoder agree on
    [eef_pos(3), eef_rot6d(6), gripper(1), base_motion(4), control_mode(1)]; "raw" stays base-first."""
    import numpy as np
    import torch

    from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import RoboCasaLeRobotDataset
    from cosmos_framework.data.generator.action.utils.action_spec import DimType
    from cosmos_framework.simulation.robocasa.eval_utils import decode_15d_armfirst_to_env12, decode_15d_to_env12

    def reader(enc):
        ds = RoboCasaLeRobotDataset.__new__(RoboCasaLeRobotDataset)
        ds._use_base_action, ds._base_encoding = True, enc
        return ds

    arm = torch.arange(1, 11, dtype=torch.float32).repeat(2, 1)  # [2, 10]: 1..10
    base = torch.full((2, 4), 20.0)
    mode = torch.full((2, 1), -1.0)
    af, rw = reader("raw_armfirst"), reader("raw")
    assert af.action_dim == rw.action_dim == 15
    a_af = af._assemble_action(arm, base, mode)
    a_rw = rw._assemble_action(arm, base, mode)
    assert a_af[0].tolist() == [*range(1, 11), 20, 20, 20, 20, -1]
    assert a_rw[0].tolist() == [20, 20, 20, 20, -1, *range(1, 11)]
    tok = torch.arange(1, 11, dtype=torch.float32)
    assert af._pad_state_token(tok).tolist() == [*range(1, 11), 0, 0, 0, 0, 0]
    assert rw._pad_state_token(tok).tolist() == [0, 0, 0, 0, 0, *range(1, 11)]
    spec = af._build_action_spec()
    assert (
        spec.types[:3] == [DimType.POS] * 3
        and spec.types[3:9] == [DimType.ROT] * 6
        and spec.types[9] == DimType.GRIPPER
    )
    assert spec.types[10:14] == [DimType.POS] * 4 and spec.types[14] == DimType.GRIPPER
    # the eval decoder inverts the layout: the same channels decode to the same env action as "raw"
    pos, rot6d, grip, cmd = [0.1, -0.2, 0.3], [1, 0, 0, 0, 1, 0], 0.7, [0.5, -0.5, 0.25, 0.0]
    for mode_val in (1.0, -1.0):
        env_af = decode_15d_armfirst_to_env12(np.array([*pos, *rot6d, grip, *cmd, mode_val]), False)
        env_rw = decode_15d_to_env12(np.array([*cmd, mode_val, *pos, *rot6d, grip]), False)
        np.testing.assert_allclose(env_af, env_rw)
        np.testing.assert_allclose(env_af[0:3], pos)
        assert env_af[6] == pytest.approx(grip) and env_af[11] == mode_val
        np.testing.assert_allclose(env_af[7:11], cmd if mode_val > 0 else [0, 0, 0, 0])


def test_reader_subsampling_is_seeded_nested_and_disjoint_from_val() -> None:
    from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import split_episode_ids
    from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import RoboCasaLeRobotDataset

    def subset(n):
        ds = RoboCasaLeRobotDataset.__new__(RoboCasaLeRobotDataset)
        ds._episodes_per_task, ds._episode_subset_seed = n, 42
        return ds

    train = split_episode_ids(total_episodes=500, seed=42, val_ratio=0.01, split="train")
    val = split_episode_ids(total_episodes=500, seed=42, val_ratio=0.01, split="val")
    assert len(train) == 495 and len(val) == 5 and not set(train) & set(val)
    meta = SimpleNamespace(root="/x/CloseFridge/20250816/lerobot")
    k50 = subset(50)._filter_valid_episodes(meta, list(train))
    k150 = subset(150)._filter_valid_episodes(meta, list(train))
    assert len(k50) == 50 and len(k150) == 150 and k50 == sorted(k50)
    assert set(k50) < set(k150) < set(train)  # nested, never a val episode
    assert k50 == subset(50)._filter_valid_episodes(meta, list(train))  # deterministic
    assert subset(None)._filter_valid_episodes(meta, list(train)) == list(train)
    assert subset(495)._filter_valid_episodes(meta, list(train)) == list(train)
    assert subset(1000)._filter_valid_episodes(meta, list(train)) == list(train)  # more than available -> keep all
    with pytest.raises(ValueError, match="episodes_per_task"):  # rejected before any shard is touched
        RoboCasaLeRobotDataset(root="/nonexistent/robocasa", episodes_per_task=0)


def test_factory_and_schema_expose_episodes_per_task() -> None:
    import inspect

    from cosmos_framework.configs.toml_config.sft_config import DataloaderTrainConfig
    from cosmos_framework.configs.toml_config.toml_config_helper import PATH_REMAPS
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_robocasa_sft_dataset
    from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import RoboCasaLeRobotDataset

    assert "episodes_per_task" in DataloaderTrainConfig.model_fields
    assert PATH_REMAPS["vfm"][("dataloader_train", "episodes_per_task")] == (
        "dataloader_train",
        "dataloader",
        "datasets",
        "robocasa",
        "dataset",
        "episodes_per_task",
    )
    assert PATH_REMAPS["vlm"][("dataloader_train", "episodes_per_task")] is None
    for fn in (get_action_robocasa_sft_dataset, RoboCasaLeRobotDataset.__init__):
        params = inspect.signature(fn).parameters
        assert params["episodes_per_task"].default is None and params["episode_subset_seed"].default == 42


def test_robocasa_policy_server_disables_the_cosmos_guardrails(tmp_path: Path) -> None:
    """Like the LIBERO server: guardrails off, otherwise OmniInference.create() tries to download the gated
    nvidia/Cosmos-Guardrail1 HF repo and the eval server dies before loading the model (seen 2026-11-01)."""
    from cosmos_framework.inference.common.args import CheckpointOverrides
    from cosmos_framework.scripts.action_policy_server_robocasa import ActionServerArgs

    config_file = tmp_path / "config.yaml"  # OmniSetupOverrides validates that the config file exists
    config_file.write_text("{}\n")
    ckpt = tmp_path / "checkpoints" / "iter_000003000"
    ckpt.mkdir(parents=True)
    args = ActionServerArgs(
        checkpoint=CheckpointOverrides.model_construct(checkpoint_path=str(ckpt), config_file=str(config_file))
    )
    overrides = args.build_setup_overrides()
    assert overrides.guardrails is False
    assert overrides.checkpoint_path == str(ckpt)


def test_launchers_and_eval_script_reference_the_recipes_with_distinct_ports() -> None:
    ex = ROOT / "examples"
    ports = {}
    for init in INITS:
        text = (ex / f"launch_sft_action_policy_robocasa_edge_{init}.sh").read_text()
        if init == "reptileinit_freshheads":
            assert "META_ACTION_INIT_PATH" not in text.replace("no META_ACTION_INIT_PATH", "")
        assert f"action_policy_robocasa_edge_{init}_${{DATA_FRACTION}}pct.toml" in text
        assert "_sft_launcher_common.sh" in text and "ROBOCASA_ROOT" in text
        found = dict(re.findall(r"^\s*(10|30|100)\)\s+_PORT=(\d+)", text, flags=re.M))
        assert sorted(found, key=int) == ["10", "30", "100"], init
        for frac, port in found.items():
            ports[(init, frac)] = int(port)
    assert len(set(ports.values())) == 9
    assert "REPTILE_CKPT_PATH" in (ex / "launch_sft_action_policy_robocasa_edge_reptileinit.sh").read_text()
    ev = (ex / "eval_robocasa_closed_loop.sh").read_text()
    for flag in (
        "--camera-set left_wrist",
        "--use-state",
        '--use-base-action --base-encoding "$BASE_ENCODING"',
        'BASE_ENCODING="${BASE_ENCODING:-raw_armfirst}"',
        "action_policy_server_robocasa",
        "summarize_results.py",
    ):
        assert flag in ev
    assert (
        'NUM_TRIALS="${NUM_TRIALS:-' in ev and 'ACTION_HORIZON="${ACTION_HORIZON:-16}"' in ev
    )  # trials are the user's call
    # bob/greco NCCL-IB guard, same as the training launcher (the 1-rank broadcast in OmniInference.create SIGSEGVs there)
    assert 'export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"' in ev
    assert (ex / "setup_robocasa365.sh").is_file()
