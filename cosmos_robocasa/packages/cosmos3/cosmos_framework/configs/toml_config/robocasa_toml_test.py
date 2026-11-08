# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""cosmos_robocasa recipes: the two ORIGINAL-RoboCasa Edge TOMLs (baseinit / reptileinit) validate, route to the
experiment, differ only in the init + head LR, carry the Cosmos Policy optimizer / schedule / augmentation, the experiment
carries the Cosmos Policy data contract on the LIBERO Edge trainable set, the reader's state-token normalization hook and
the 12D/16D converter layouts round-trip through the reader, and the launchers / setup script reference the right files."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import tomllib
import torch

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

ROOT = Path(__file__).resolve().parents[3]
TOML_DIR = ROOT / "examples/toml/sft_config"
INITS = ("baseinit", "reptileinit")
_AUG_NODE = "dataloader_train.dataloader.datasets.robocasa.dataset.use_image_augmentation"
STATS_JSON = ROOT / "cosmos_framework/data/generator/action/normalizer_stats/robocasa_cosmos_policy_frame_wise_relative_rot6d.json"
GLOBAL_BATCH = 512


def _toml(init: str) -> dict:
    return tomllib.load(open(TOML_DIR / f"action_policy_robocasa_edge_{init}.toml", "rb"))


def _global_batch(raw: dict) -> int:
    par = raw["model"]["parallelism"]
    assert par.get("data_parallel_replicate_degree", 1) == 1
    return par["data_parallel_shard_degree"] * raw["dataloader_train"]["max_samples_per_batch"] * raw["trainer"].get(
        "grad_accum_iter", 1
    )


@pytest.mark.parametrize("init", INITS)
def test_toml_validates_and_routes(init: str) -> None:
    raw = _toml(init)
    cfg = SFTExperimentConfig.model_validate(raw)
    overrides = build_hydra_overrides(raw)
    assert "experiment=action_policy_robocasa_edge" in overrides
    assert cfg.job.name == f"edge_robocasa_h50_fullft_{init}" and cfg.job.project == "cosmos3_action_robocasa"
    assert _global_batch(raw) == GLOBAL_BATCH
    assert raw["model"]["lora_enabled"] is False and raw["model"]["ema"]["enabled"] is True
    # Cosmos Policy optimizer (Predict2 defaults): AdamW 1e-4, betas (0.9, 0.99), eps 1e-8, wd 0.1
    opt = raw["optimizer"]
    assert opt["lr"] == 1e-4 and opt["betas"] == [0.9, 0.99] and opt["eps"] == 1e-8 and opt["weight_decay"] == 0.1
    # Cosmos Policy 2-cycle schedule: warmup -> linear decay to 0.3x at ~2/3 of the budget -> 0.06x constant
    sch = raw["scheduler"]
    assert len(sch["cycle_lengths"]) == len(sch["warm_up_steps"]) == len(sch["f_start"]) == len(sch["f_max"]) == 2
    assert sch["f_max"][0] == 1.0 and sch["f_min"][0] == 0.3 and sch["f_start"][0] < 1e-3
    assert sch["f_start"][1] == sch["f_max"][1] == sch["f_min"][1] == 0.06 and sch["warm_up_steps"][1] == 0
    assert 0 < sch["warm_up_steps"][0] < sch["cycle_lengths"][0] < raw["trainer"]["max_iter"] < sch["cycle_lengths"][1]
    assert abs(sch["cycle_lengths"][0] / raw["trainer"]["max_iter"] - 2 / 3) < 0.05
    # "stronger" augmentation on, validation off (in-sample monitor only)
    assert raw["dataloader_train"]["use_image_augmentation"] is True
    assert f"{_AUG_NODE}=True" in overrides or f"{_AUG_NODE}=true" in overrides
    assert not any(o.startswith("dataloader_val") and "use_image_augmentation" in o for o in overrides)
    assert raw["trainer"]["run_validation"] is False and raw["trainer"]["run_validation_on_start"] is False
    assert raw["checkpoint"]["save_iter"] <= raw["trainer"]["max_iter"]


def test_baseinit_and_reptileinit_differ_only_in_init_and_head_lr() -> None:
    base, rep = _toml("baseinit"), _toml("reptileinit")
    for section in ("model", "scheduler", "trainer", "dataloader_train", "dataloader_val"):
        assert base[section] == rep[section], section
    assert {k: v for k, v in base["job"].items() if k != "name"} == {k: v for k, v in rep["job"].items() if k != "name"}
    b_opt, r_opt = copy.deepcopy(base["optimizer"]), copy.deepcopy(rep["optimizer"])
    assert b_opt.pop("lr_multipliers") == {"action2llm": 5.0, "llm2action": 5.0, "action_modality_embed": 5.0}
    assert r_opt.pop("lr_multipliers") == {"action2llm": 1.0, "llm2action": 1.0, "action_modality_embed": 1.0}
    assert b_opt == r_opt
    # init
    assert base["checkpoint"]["load_path"] == "${oc.env:BASE_CHECKPOINT_PATH}"
    assert "meta_action_init_path" not in base["checkpoint"] and "keys_to_skip_loading" not in base["checkpoint"]
    assert rep["checkpoint"]["load_path"] == "${oc.env:REPTILE_CKPT_PATH}"
    assert rep["checkpoint"]["meta_action_init_path"] == "${oc.env:META_ACTION_INIT_PATH}"
    assert rep["checkpoint"]["meta_action_init_include_lora"] is False
    assert rep["checkpoint"]["keys_to_skip_loading"] == ["net_ema.", "action2llm", "llm2action", "action_modality_embed"]
    from cosmos_framework.data.generator.action.utils.domain_utils import EMBODIMENT_TO_DOMAIN_ID

    assert rep["checkpoint"]["meta_action_init_domain_id"] == EMBODIMENT_TO_DOMAIN_ID["robocasa"] == 30


def test_experiment_carries_the_cosmos_policy_contract_on_the_libero_edge_trainable_set(monkeypatch) -> None:
    import cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_edge as libero
    import cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_robocasa_edge as m
    from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import (
        DEFAULT_ROBOCASA_COSMOS_POLICY_TASKS,
    )

    monkeypatch.setenv("ROBOCASA_ROOT", "/data/robocasa_cosmos_policy_lerobot_v3")  # the LazyDict resolves ${oc.env:...}
    exp = m.action_policy_robocasa_edge
    assert exp["optimizer"]["keys_to_select"] == libero.action_policy_libero_edge["optimizer"]["keys_to_select"]
    assert "k_norm_und_for_gen" in exp["optimizer"]["keys_to_select"]
    assert exp["model"]["config"] == libero.action_policy_libero_edge["model"]["config"]
    assert exp["checkpoint"]["keys_to_skip_loading"] == libero.action_policy_libero_edge["checkpoint"]["keys_to_skip_loading"]
    assert exp["optimizer"]["lr"] == 1e-4 and exp["optimizer"]["weight_decay"] == 0.1
    assert exp["optimizer"]["betas"] == [0.9, 0.99] and exp["optimizer"]["eps"] == 1e-8
    assert len(DEFAULT_ROBOCASA_COSMOS_POLICY_TASKS) == len(set(DEFAULT_ROBOCASA_COSMOS_POLICY_TASKS)) == 24
    for loader, split, ratio, n_workers, aug, cfg in (
        ("dataloader_train", "train", 0.0, 4, True, 0.0),
        ("dataloader_val", "val", 0.05, 2, False, 0.0),
    ):
        node = exp[loader]["dataloader"]["datasets"]["robocasa"]["dataset"]
        assert node["root"] == "/data/robocasa_cosmos_policy_lerobot_v3"
        assert tuple(node["task_names"]) == DEFAULT_ROBOCASA_COSMOS_POLICY_TASKS
        assert node["use_base_action"] is False  # 10D [pos, rot6d, gripper]: the Cosmos Policy 7D arm command
        assert node["camera_set"] == "left_wrist" and node["use_state"] is True
        assert node["chunk_length"] == 16 and node["fps"] == 20
        assert node["action_normalization"] == "quantile_rot" and node["state_normalization"] == "minmax"
        assert node["split"] == split and node["split_val_ratio"] == ratio and node["split_seed"] == 42
        assert node["episodes_per_task"] is None and node["iterable_shuffle"] is True
        assert node["use_image_augmentation"] is aug and node["cfg_dropout_rate"] == cfg  # no text dropout (Cosmos Policy)
        assert node["augment_rotation_deg"] == 5.0 and node["augment_hue"] == 0.05 and node["augment_crop_scale"] == 0.95
        assert node["format_prompt_as_json"] is True and node["resolution"] is None
        assert exp[loader]["dataloader"]["num_workers"] == n_workers
    assert "val_loss_breakdown" in exp["trainer"]["callbacks"]


def test_reader_state_token_hook_normalizes_row0_with_its_own_stats() -> None:
    """``_normalize_action_rows``: row 0 (state token) -> state normalizer, rows 1: -> action normalizer."""
    from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import RoboCasaLeRobotDataset
    from cosmos_framework.data.generator.action.utils.action_processing import resolve_action_normalization

    ds = object.__new__(RoboCasaLeRobotDataset)
    d = 10
    act_stats = {k: torch.full((d,), v) for k, v in {"q01": -2.0, "q99": 2.0}.items()}
    st_stats = {k: torch.full((d,), v) for k, v in {"min": 0.0, "max": 10.0}.items()}
    ds._action_normalizer = resolve_action_normalization("quantile_rot", act_stats)  # (x - 0) / 2
    ds._state_normalizer = resolve_action_normalization("minmax", st_stats)  # (x - 5) / 5
    ds._use_state = True
    action = torch.cat([torch.full((1, d), 10.0), torch.full((3, d), 1.0)], dim=0)  # token row + 3 delta rows
    out = ds._normalize_action_rows(action)
    torch.testing.assert_close(out[0], torch.full((d,), 1.0))  # (10 - 5) / 5
    torch.testing.assert_close(out[1:], torch.full((3, d), 0.5))  # 1 / 2
    ds._state_normalizer = None
    torch.testing.assert_close(ds._normalize_action_rows(action), action / 2.0)  # plain path = the base behaviour


def test_reader_rejects_unnormalized_state_token_with_normalized_actions(tmp_path: Path) -> None:
    from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import RoboCasaLeRobotDataset

    ds = object.__new__(RoboCasaLeRobotDataset)
    ds._action_stats_path = str(tmp_path / "x.json")
    assert ds._normalizer_path() == tmp_path / "x.json"  # absolute path wins
    ds._action_stats_path = "foo.json"
    assert ds._normalizer_path() == RoboCasaLeRobotDataset._ROBOCASA_NORMALIZERS_DIR / "foo.json"
    ds._action_stats_path = None
    assert ds._normalizer_path().name == "robocasa_cosmos_policy_frame_wise_relative_rot6d.json"
    assert ds._normalizer_path().parent.name == "normalizer_stats"
    with pytest.raises((ValueError, FileNotFoundError)):
        RoboCasaLeRobotDataset(
            root=str(tmp_path),
            use_state=True,
            action_normalization="quantile_rot",
            state_normalization=None,
            action_stats_path=str(tmp_path / "missing.json"),
        )


def test_converter_layouts_round_trip_through_the_reader() -> None:
    """build_action12 / build_state16 put the 7D arm command and the 9D robot_states exactly where the reader slices."""
    import numpy as np

    from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import RoboCasaLeRobotDataset
    from cosmos_framework.scripts.convert_robocasa_cosmos_policy_to_lerobot_v30 import (
        build_action12,
        build_state16,
        compute_stats_blocks,
        reader_action10,
        reader_state10,
    )

    rng = np.random.default_rng(0)
    t = 5
    act7 = rng.uniform(-1, 1, size=(t, 7)).astype(np.float32)
    act7[:, 6] = np.sign(act7[:, 6])
    rs9 = rng.normal(size=(t, 9)).astype(np.float32)
    rs9[:, 5:9] /= np.linalg.norm(rs9[:, 5:9], axis=-1, keepdims=True)  # unit quaternions
    a12, s16 = build_action12(act7), build_state16(rs9)
    assert a12.shape == (t, 12) and s16.shape == (t, 16)
    assert (a12[:, :4] == 0).all() and (a12[:, 4] == -1).all() and (s16[:, :6] == 0).all() and (s16[:, 6] == 1).all()
    ds = object.__new__(RoboCasaLeRobotDataset)
    ds._chunk_length = t - 1
    via_reader = RoboCasaLeRobotDataset._build_frame_wise_action(ds, torch.from_numpy(a12))
    torch.testing.assert_close(via_reader, torch.from_numpy(reader_action10(act7)))
    torch.testing.assert_close(via_reader[:, :3], torch.from_numpy(act7[:, :3]))
    torch.testing.assert_close(via_reader[:, 9], torch.from_numpy(act7[:, 6]))
    token = RoboCasaLeRobotDataset._build_initial_state(ds, torch.from_numpy(s16))  # frame 0 of the chunk+1 window
    torch.testing.assert_close(token, torch.from_numpy(reader_state10(rs9))[0])
    blocks = compute_stats_blocks(reader_action10(act7), reader_state10(rs9))
    assert set(blocks) == {"global", "global_raw", "state"}
    for block in blocks.values():
        assert set(block) == {"mean", "std", "min", "max", "q01", "q99"} and all(len(v) == 10 for v in block.values())
    for d in (3, 4, 5, 6, 7, 8):  # rot6d dims: identity in "global", real stats in "global_raw"
        assert blocks["global"]["q01"][d] == -1.0 and blocks["global"]["q99"][d] == 1.0
    assert blocks["global_raw"]["q99"][0] == pytest.approx(float(np.quantile(act7[:, 0], 0.99)), abs=1e-5)


def test_bundled_stats_json_matches_the_reader_contract() -> None:
    if not STATS_JSON.exists():
        pytest.skip(f"{STATS_JSON.name} not generated yet (STEPS=stats examples/setup_robocasa.sh)")
    import json

    d = json.loads(STATS_JSON.read_text())
    assert d["metadata"]["action_dim"] == 10 and d["metadata"]["state_dim"] == 10
    for block in ("global", "global_raw", "state"):
        assert all(len(d[block][k]) == 10 for k in ("mean", "std", "min", "max", "q01", "q99")), block
        assert all(lo <= hi for lo, hi in zip(d[block]["q01"], d[block]["q99"], strict=True)), block
    assert d["metadata"]["num_shards"] == 24 and d["metadata"]["num_frames"] > 100_000


def test_launchers_and_setup_reference_the_recipes_with_distinct_ports() -> None:
    ex = ROOT / "examples"
    ports = {}
    for init in INITS:
        text = (ex / f"launch_sft_action_policy_robocasa_edge_{init}.sh").read_text()
        assert f"action_policy_robocasa_edge_{init}.toml" in text
        assert "_sft_launcher_common.sh" in text and "ROBOCASA_ROOT" in text and "robocasa_cosmos_policy_lerobot_v3" in text
        assert "robocasa_cosmos_policy_frame_wise_relative_rot6d.json" in text  # the stats the recipe needs
        assert "DATA_FRACTION" not in text
        m = re.search(r'MASTER_PORT:=(\d+)', text)
        assert m, init
        ports[init] = int(m.group(1))
    assert len(set(ports.values())) == 2
    rep = (ex / "launch_sft_action_policy_robocasa_edge_reptileinit.sh").read_text()
    assert "REPTILE_CKPT_PATH" in rep and "META_ACTION_INIT_PATH" in rep
    assert "META_ACTION_INIT_PATH" not in (ex / "launch_sft_action_policy_robocasa_edge_baseinit.sh").read_text()
    setup = (ex / "setup_robocasa.sh").read_text()
    for s in (
        "nvidia/RoboCasa-Cosmos-Policy",
        "convert_robocasa_cosmos_policy_to_lerobot_v30",
        "--stats-only",
        "robocasa_cosmos_policy_frame_wise_relative_rot6d.json",
    ):
        assert s in setup
    # Reptile meta-training pair kept: the TOML validates (custom section) and its launcher points at it
    v4 = tomllib.load(open(TOML_DIR / "action_reptile_meta_edge_v4.toml", "rb"))
    assert SFTExperimentConfig.model_validate(v4).custom["meta"]["theta_mode"] == "full"
    assert v4["job"]["experiment"] == "action_reptile_meta_edge"
    assert "action_reptile_meta_edge_v4.toml" in (ex / "launch_reptile_meta_edge_v4.sh").read_text()
    assert sorted(p.name for p in TOML_DIR.glob("*.toml")) == [
        "action_policy_robocasa_edge_baseinit.toml",
        "action_policy_robocasa_edge_reptileinit.toml",
        "action_reptile_meta_edge_v4.toml",
    ]
