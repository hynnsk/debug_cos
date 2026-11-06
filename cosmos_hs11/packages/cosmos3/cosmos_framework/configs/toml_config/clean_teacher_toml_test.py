# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""cosmos_hs11 v126-v137: REPA on the CLEAN predicted frames from a frozen-network teacher (self_init / dcp_init with
teacher_input='clean_context') or DINOv2, on top of the ranking recipe v111 -- structure, routing and launchers."""

from __future__ import annotations

import copy
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig
from cosmos_framework.configs.toml_config.toml_config_helper import build_hydra_overrides

_TOML_DIR = Path(__file__).resolve().parents[3] / "examples" / "toml" / "sft_config"
_EX = _TOML_DIR.parents[1]
_STEM = "action_policy_libero_10_edge_reptileinit"
# version: (teacher, student block, teacher block, feature w, relation w)
_ARMS = {
    126: ("self_init", 14, 14, 5.0, 0.0),
    127: ("self_init", 24, 24, 5.0, 0.0),
    128: ("self_init", 14, 14, 0.0, 5.0),
    129: ("self_init", 24, 24, 0.0, 5.0),
    130: ("dcp_init", 14, 14, 5.0, 0.0),
    131: ("dcp_init", 24, 24, 5.0, 0.0),
    132: ("dcp_init", 14, 14, 0.0, 5.0),
    133: ("dcp_init", 24, 24, 0.0, 5.0),
    134: ("dinov2_vitb14", 14, None, 5.0, 0.0),
    135: ("dinov2_vitb14", 24, None, 5.0, 0.0),
    136: ("dinov2_vitb14", 14, None, 0.0, 5.0),
    137: ("dinov2_vitb14", 24, None, 0.0, 5.0),
}
_PORTS = {v: 50847 + i for i, v in enumerate(sorted(_ARMS))}


def _expected_repa(teacher: str, layer: int, teacher_layer: int | None, w: float, rw: float) -> dict:
    repa = {
        "enabled": True, "teacher": teacher, "layer_index": layer, "objective": "token", "projector_type": "mlp",
        "loss_weight": w, "relation_loss_weight": rw, "relation_distance": "l1", "relation_margin": 0.1,
    }
    if teacher == "dinov2_vitb14":
        repa.update({"teacher_input_size": 224, "target_adapter": "avgpool", "target_grid_thw": [4, 5, 5]})
    else:
        repa.update({"teacher_layer_index": teacher_layer, "teacher_input": "clean_context"})
        if teacher == "dcp_init":
            repa["teacher_checkpoint_path"] = "${oc.env:BASE_CHECKPOINT_PATH}"
    return repa


@pytest.mark.parametrize("version", sorted(_ARMS))
def test_clean_teacher_recipe_is_v111_plus_the_repa_block(version: int) -> None:
    teacher, layer, teacher_layer, w, rw = _ARMS[version]
    base = tomllib.load(open(_TOML_DIR / f"{_STEM}_v111.toml", "rb"))
    raw = tomllib.load(open(_TOML_DIR / f"{_STEM}_v{version}.toml", "rb"))
    repa = _expected_repa(teacher, layer, teacher_layer, w, rw)
    assert raw["model"]["repa"] == repa
    expected = copy.deepcopy(base)
    expected["job"]["name"] = f"edge_libero10_3ep_fullft_reptile_v{version}"
    expected["job"]["experiment"] = "action_policy_libero_edge_repa"
    expected["model"]["repa"] = repa
    expected["checkpoint"]["keys_to_skip_loading"] = ["net_ema.", "repa_", "action2llm", "llm2action", "action_modality_embed"]
    assert raw == expected  # nothing else differs from v111 (ranking hinge relative 0.3 / w 1.0, fresh heads, bs 128, 2000 iters)
    assert raw["model"]["language_ranking"] == base["model"]["language_ranking"]
    assert "meta_action_init_path" not in raw["checkpoint"]
    cfg = SFTExperimentConfig.model_validate(raw)
    assert cfg.model.repa.teacher == teacher and cfg.model.repa.layer_index == layer
    overrides = build_hydra_overrides(raw)
    assert "experiment=action_policy_libero_edge_repa" in overrides
    assert f"model.config.repa.teacher={teacher}" in overrides and f"model.config.repa.layer_index={layer}" in overrides
    assert f"model.config.repa.loss_weight={w}" in overrides and f"model.config.repa.relation_loss_weight={rw}" in overrides
    assert "model.config.repa.relation_margin=0.1" in overrides and "model.config.repa.relation_distance=l1" in overrides
    if teacher == "dinov2_vitb14":
        assert not any(o.startswith(("model.config.repa.teacher_layer_index", "model.config.repa.teacher_input=")) for o in overrides)
        assert "model.config.repa.teacher_input_size=224" in overrides
    else:
        assert f"model.config.repa.teacher_layer_index={teacher_layer}" in overrides
        assert "model.config.repa.teacher_input=clean_context" in overrides
    if teacher == "dcp_init":
        assert any(o.startswith("model.config.repa.teacher_checkpoint_path=") and "BASE_CHECKPOINT_PATH" in o for o in overrides)
    else:
        assert not any(o.startswith("model.config.repa.teacher_checkpoint_path") for o in overrides)


@pytest.mark.parametrize("version", sorted(_ARMS))
def test_clean_teacher_launcher_drives_its_toml(version: int) -> None:
    teacher = _ARMS[version][0]
    launcher = _EX / f"launch_sft_{_STEM}_v{version}.sh"
    text = launcher.read_text()
    assert f"{_STEM}_v{version}.toml" in text and f'MASTER_PORT:={_PORTS[version]}' in text
    assert "launch_reptile_meta_edge.sh first" in text  # REPTILE_CKPT_PATH/model check inherited from v111
    if teacher == "dcp_init":
        # the frozen teacher's checkpoint must be given explicitly and must not be the Reptile init
        assert 'export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-}"' in text
        assert ': "${BASE_CHECKPOINT_PATH:=${REPTILE_CKPT_PATH}}"' not in text
        assert '[[ -d "$BASE_CHECKPOINT_PATH/model" ]]' in text and "equals REPTILE_CKPT_PATH" in text
    else:
        assert ': "${BASE_CHECKPOINT_PATH:=${REPTILE_CKPT_PATH}}"' in text
    subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_clean_teacher_launcher_ports_are_unique() -> None:
    ports: dict[int, list[str]] = {}
    for launcher in _EX.glob("launch_sft_*.sh"):
        m = re.search(r"MASTER_PORT[:=]+(\d+)", launcher.read_text())
        if m:
            ports.setdefault(int(m.group(1)), []).append(launcher.name)
    for version in _ARMS:
        name = f"launch_sft_{_STEM}_v{version}.sh"
        assert ports[_PORTS[version]] == [name], (name, ports[_PORTS[version]])
