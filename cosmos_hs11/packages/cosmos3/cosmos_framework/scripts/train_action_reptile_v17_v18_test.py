# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for the cosmos_hs11 v17 / v18 trainer knobs: ``[custom.meta]`` validation (``private_lora``,
``query_steps``), the ranking loader auto-switch, the ranking scalars, the task-text lookup and the raw-row extras
path of ``ActionSFTDataset``. Every other recipe must stay bit-identical (both knobs off by default)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
from cosmos_framework.data.generator.action.meta.embodiments import embodiment_task_texts
from cosmos_framework.scripts.train_action_meta import SAMPLER_OVERRIDE_KEYS
from cosmos_framework.scripts.train_action_reptile import (
    _RANKING_LOG_KEYS,
    ReptileTrainConfig,
    _ranking_scalars,
    apply_ranking_loader_settings,
)


def test_config_defaults_and_validation() -> None:
    cfg = ReptileTrainConfig.from_dict({})
    assert cfg.private_lora is False and cfg.query_steps == 0  # every existing recipe
    assert ReptileTrainConfig.from_dict({"private_lora": True}).private_lora is True
    assert ReptileTrainConfig.from_dict({"query_steps": 2}).query_steps == 2
    with pytest.raises(ValueError, match="query_steps"):
        ReptileTrainConfig.from_dict({"query_steps": -1})
    with pytest.raises(ValueError, match="private_lora"):
        ReptileTrainConfig.from_dict({"private_lora": True, "theta_mode": "lora"})
    assert "language_ranking" in SAMPLER_OVERRIDE_KEYS and "query_steps" not in SAMPLER_OVERRIDE_KEYS


def _config(enabled: bool, loader: bool | None = False):
    dl = {} if loader is None else {"language_ranking": loader}
    return SimpleNamespace(model=SimpleNamespace(config={"language_ranking": {"enabled": enabled}}), dataloader_train=dl)


def test_apply_ranking_loader_settings() -> None:
    c = _config(True)
    assert apply_ranking_loader_settings(c) == {"language_ranking": True} and c.dataloader_train == {"language_ranking": True}
    c = _config(True, loader=True)  # explicit [custom.meta] value: nothing to do
    assert apply_ranking_loader_settings(c) == {}
    c = _config(False)  # ranking off: untouched (every existing recipe)
    assert apply_ranking_loader_settings(c) == {} and c.dataloader_train == {"language_ranking": False}
    assert apply_ranking_loader_settings(SimpleNamespace(model=SimpleNamespace(config={}), dataloader_train={})) == {}
    with pytest.raises(KeyError):
        apply_ranking_loader_settings(_config(True, loader=None))


def test_ranking_scalars_pick_the_logged_terms() -> None:
    out = {
        "language_ranking_loss": torch.tensor(0.25),
        "language_ranking_gap": 0.5,
        "language_ranking_margin_satisfied": torch.tensor(1.0),
        "language_ranking_valid_fraction": torch.tensor(0.75),
        "flow_matching_loss_action": torch.tensor(2.0),
    }
    assert _ranking_scalars(out) == {
        "language_ranking_loss": 0.25,
        "language_ranking_gap": 0.5,
        "language_ranking_margin_satisfied": 1.0,
        "language_ranking_valid_fraction": 0.75,
    }
    assert _ranking_scalars({"repa_loss": torch.tensor(1.0)}) == {}
    assert set(_RANKING_LOG_KEYS) == {
        "language_ranking_loss",
        "language_ranking_gap",
        "language_ranking_margin_satisfied",
        "language_ranking_valid_fraction",
    }


def test_embodiment_task_texts_reads_the_reader_task_table() -> None:
    assert embodiment_task_texts(SimpleNamespace(_tasks=["open", "close | shut"])) == {0: "open", 1: "close | shut"}
    assert embodiment_task_texts(SimpleNamespace(_dataset=SimpleNamespace(_tasks={3: "pick", 7: "place"}))) == {3: "pick", 7: "place"}
    # the YAM repo collection keeps a GLOBAL task table as ``task_texts`` (no ``_tasks``)
    assert embodiment_task_texts(SimpleNamespace(_dataset=SimpleNamespace(task_texts=["fold towel", "stack cups"]))) == {0: "fold towel", 1: "stack cups"}
    assert embodiment_task_texts(SimpleNamespace(_tasks={0: "a"}, task_texts=["ignored"])) == {0: "a"}  # _tasks wins
    assert embodiment_task_texts(SimpleNamespace(_dataset=SimpleNamespace())) is None


def test_action_sft_dataset_getitem_with_extras_merges_into_the_raw_row() -> None:
    seen = []

    class _Inner:
        def __getitem__(self, i):
            return {"ai_caption": f"instruction {i}", "video": i}

    def _transform(raw, resolution):
        seen.append((dict(raw), resolution))
        return {"ok": raw.get("ranking_negative_instruction")}

    ds = ActionSFTDataset(_Inner(), _transform, "256")
    assert ds[1] == {"ok": None} and seen[-1][0] == {"ai_caption": "instruction 1", "video": 1}  # plain path untouched
    assert ds.getitem_with_extras(2, {"ranking_negative_instruction": "close it"}) == {"ok": "close it"}
    assert seen[-1][0] == {"ai_caption": "instruction 2", "video": 2, "ranking_negative_instruction": "close it"}
    # None = no eligible negative: the sample's own instruction stands in (the loader then masks the window)
    assert ds.getitem_with_extras(3, {"ranking_negative_instruction": None}) == {"ok": "instruction 3"}
    assert ds.getitem_with_extras(4, None) == {"ok": None} and ds.getitem_with_extras(5, {}) == {"ok": None}
