# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""cosmos_hs11 v115-v123: the internal-teacher REPA recipes (self-distillation from the frozen initial network, Wan VAE
latent patches) on top of the fresh-head LIBERO-10 recipe v108 -- structure, routing and launchers."""

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
_ARMS = {
    115: ("self_init", 28, "mlp", 5.0, 0.0),
    116: ("self_init", 28, "mlp", 0.0, 5.0),
    117: ("self_init", 28, "mlp", 5.0, 5.0),
    118: ("self_init", 14, "mlp", 5.0, 0.0),
    119: ("self_init", 14, "mlp", 0.0, 5.0),
    120: ("self_init", 14, "mlp", 5.0, 5.0),
    121: ("wan_vae", None, "mlp", 5.0, 0.0),
    122: ("wan_vae", None, "mlp", 0.0, 5.0),
    123: ("wan_vae", None, "mlp", 5.0, 5.0),
}


@pytest.mark.parametrize("version", sorted(_ARMS))
def test_internal_teacher_recipe_is_v108_plus_the_repa_block(version: int) -> None:
    teacher, teacher_layer, projector, w, rw = _ARMS[version]
    base = tomllib.load(open(_TOML_DIR / f"{_STEM}_v108.toml", "rb"))
    raw = tomllib.load(open(_TOML_DIR / f"{_STEM}_v{version}.toml", "rb"))
    repa = {
        "enabled": True, "teacher": teacher, "layer_index": 14, "objective": "token", "projector_type": projector,
        "loss_weight": w, "relation_loss_weight": rw, "relation_distance": "l1", "relation_margin": 0.1,
    }
    if teacher_layer is not None:
        repa["teacher_layer_index"] = teacher_layer
    assert raw["model"]["repa"] == repa
    expected = copy.deepcopy(base)
    expected["job"]["name"] = f"edge_libero10_3ep_fullft_reptile_v{version}"
    expected["job"]["experiment"] = "action_policy_libero_edge_repa"
    expected["model"]["repa"] = repa
    expected["checkpoint"]["keys_to_skip_loading"] = ["net_ema.", "repa_", "action2llm", "llm2action", "action_modality_embed"]
    assert raw == expected  # nothing else differs from v108 (fresh heads, head LR x5, 2000 iters, bs 128, wandb offline)
    assert "meta_action_init_path" not in raw["checkpoint"] and raw["job"]["wandb_mode"] == "offline"
    cfg = SFTExperimentConfig.model_validate(raw)
    assert cfg.model.repa.teacher == teacher and cfg.model.repa.teacher_layer_index == teacher_layer
    overrides = build_hydra_overrides(raw)
    assert "experiment=action_policy_libero_edge_repa" in overrides
    assert f"model.config.repa.teacher={teacher}" in overrides and "model.config.repa.layer_index=14" in overrides
    assert f"model.config.repa.projector_type={projector}" in overrides
    assert f"model.config.repa.loss_weight={w}" in overrides and f"model.config.repa.relation_loss_weight={rw}" in overrides
    assert "model.config.repa.relation_distance=l1" in overrides
    if teacher_layer is None:
        assert not any(o.startswith("model.config.repa.teacher_layer_index") for o in overrides)
    else:
        assert f"model.config.repa.teacher_layer_index={teacher_layer}" in overrides
    launcher = _EX / f"launch_sft_{_STEM}_v{version}.sh"
    text = launcher.read_text()
    assert f"{_STEM}_v{version}.toml" in text
    subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_internal_teacher_launcher_ports_are_unique() -> None:
    ports = {}
    for launcher in _EX.glob("launch_sft_*.sh"):
        m = re.search(r"MASTER_PORT[:=]+(\d+)", launcher.read_text())
        if m:
            ports.setdefault(int(m.group(1)), []).append(launcher.name)
    for version in _ARMS:
        name = f"launch_sft_{_STEM}_v{version}.sh"
        port = next(p for p, names in ports.items() if name in names)
        assert ports[port] == [name], (name, ports[port])
