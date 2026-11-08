# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import copy
import json
import random
from collections import Counter

import pytest
import torch

from cosmos_framework.data.generator.action.utils.language_ranking import LiberoInstructionNegatives
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.augmentors.text_tokenizer import TextTokenizerTransform


def test_negatives_are_uniform_over_other_nine_tasks_and_reproducible():
    tasks = {i * 3: f"task {i}" for i in range(10)}  # task IDs need not be contiguous
    sampler = LiberoInstructionNegatives(tasks)
    state = random.getstate()
    try:
        random.seed(123)
        draws = [sampler.sample(9) for _ in range(9000)]
        counts = Counter(draws)
        assert set(counts) == set(tasks.values()) - {tasks[9]}
        assert all(850 < n < 1150 for n in counts.values())
        random.seed(123)
        assert draws[:30] == [sampler.sample(9) for _ in range(30)]
    finally:
        random.setstate(state)


def test_rejects_wrong_suite_and_empty_instruction():
    with pytest.raises(ValueError, match="exactly 10"):
        LiberoInstructionNegatives({0: "one", 1: "two"})
    with pytest.raises(ValueError, match="nonempty"):
        LiberoInstructionNegatives({i: "task" if i else "" for i in range(10)})


class CharacterProcessor:
    def tokenize_text(self, text, **kwargs):
        return [ord(c) for c in text] or [0]


@pytest.mark.parametrize("dropout", [0.0, 1.0])
def test_transform_and_packing_preserve_metadata_and_cfg_dropout(monkeypatch, dropout):
    from cosmos_framework.data.generator.action.meta.episodic_sampler import pack_samples_into_batch
    from cosmos_framework.data.generator.augmentors import text_tokenizer
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    monkeypatch.setattr(text_tokenizer, "lazy_instantiate", lambda cfg: CharacterProcessor())
    transform = ActionTransformPipeline(tokenizer_config=None, format_prompt_as_json=True)
    transform.text_tokenizer = TextTokenizerTransform(
        input_keys=["ai_caption"],
        output_keys=["text_token_ids"],
        args={"tokenizer_config": {}, "cfg_dropout_rate": dropout},
    )
    sample = dict(
        video=torch.zeros(3, 17, 256, 512, dtype=torch.uint8),
        action=torch.zeros(16, 10),
        ai_caption="pick the bowl",
        mode="wam",
        viewpoint="concat_view",
        conditioning_fps=20.0,
        domain_id=5,
    )
    baseline = transform(copy.deepcopy(sample), None)
    sample["ranking_negative_instruction"] = "put the plate in the drawer"
    ranked = transform(copy.deepcopy(sample), None)
    torch.testing.assert_close(baseline["text_token_ids"], ranked["text_token_ids"])
    torch.testing.assert_close(baseline["action"], ranked["action"])
    torch.testing.assert_close(baseline["video"], ranked["video"])
    assert ranked["ranking_valid"] is (dropout == 0)
    assert "ranking_negative_instruction" not in ranked
    neg = json.loads("".join(chr(i) for i in ranked["negative_text_token_ids"].tolist()))
    assert neg["actions"][0]["description"] == "put the plate in the drawer."
    if dropout == 0:
        pos = json.loads(ranked["ai_caption"])
        neg["actions"][0]["description"] = pos["actions"][0]["description"]
        assert neg == pos
    packed = pack_samples_into_batch([ranked, copy.deepcopy(ranked)], "libero")
    neg_tokens = OmniMoTModel._load_and_tokenize_text_data(
        None, {"text_token_ids": packed["negative_text_token_ids"]}, 0
    )
    assert neg_tokens == [ranked["negative_text_token_ids"].tolist()] * 2
    assert [bool(v.item()) for v in packed["ranking_valid"]] == [dropout == 0] * 2

    # Exercise the production input preparation and CP payload round-trip with a stand-in VAE.
    from types import MethodType, SimpleNamespace

    from cosmos_framework.configs.base.defaults.model_config import LanguageRankingConfig
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

    clean = GenerationDataClean(
        batch_size=2,
        is_image_batch=False,
        raw_state_vision=[torch.zeros(1, 3, 17, 4, 8)] * 2,
        x0_tokens_vision=[torch.zeros(1, 3, 5, 4, 8)] * 2,
        x0_tokens_action=[torch.zeros(16, 64)] * 2,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(language_ranking=LanguageRankingConfig(enabled=True)),
        input_video_key="video",
        input_image_key="images",
        get_data_and_condition=lambda *args, **kwargs: clean,
    )
    for method in ("memory_init_training", "_load_and_tokenize_text_data"):
        setattr(model, method, MethodType(getattr(OmniMoTModel, method), model))
    inputs = OmniMoTModel._prepare_training_data(model, packed, 0)
    restored = OmniMoTModel._unpack_training_payload(OmniMoTModel._pack_training_payload(*inputs))
    assert restored[3]["ranking_negative_text_indexes"] == neg_tokens
    assert restored[3]["ranking_valid"] == [dropout == 0] * 2


# ---------------------------------------------------------------------------------------------------------------
# cosmos_hs11 v27: weighted (action-similarity) hard negatives
# ---------------------------------------------------------------------------------------------------------------
def test_weighted_negatives_follow_the_weights_and_validate_them():
    import numpy as np

    tasks = {i: f"task {i}" for i in range(10)}
    weights = {i: [1.0] * 9 for i in range(10)}
    weights[0] = [0.0] * 9
    weights[0][0] = 3.0  # candidates of task 0 are tasks 1..9 -> all mass on task 1 (unnormalized on purpose)
    weights[5] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # candidates of 5: 0,1,2,3,4,6,7,8,9 -> tasks 0 and 9
    sampler = LiberoInstructionNegatives(tasks, weights=weights)
    assert sampler.weights[0] == [1.0] + [0.0] * 8
    state = random.getstate()
    try:
        random.seed(5)
        assert {sampler.sample(0) for _ in range(200)} == {"task 1"}
        counts = Counter(sampler.sample(5) for _ in range(4000))
        assert set(counts) == {"task 0", "task 9"} and all(1800 < n < 2200 for n in counts.values())
        counts = Counter(sampler.sample(3) for _ in range(9000))  # uniform row -> uniform draws
        assert set(counts) == set(tasks.values()) - {"task 3"} and all(850 < n < 1150 for n in counts.values())
    finally:
        random.setstate(state)
    assert LiberoInstructionNegatives(tasks).weights is None
    with pytest.raises(ValueError, match="exactly the dataset's task ids"):
        LiberoInstructionNegatives(tasks, weights={i: [1.0] * 9 for i in range(1, 11)})
    with pytest.raises(ValueError, match="expected 9"):
        LiberoInstructionNegatives(tasks, weights={**weights, 2: [1.0] * 8})
    with pytest.raises(ValueError, match="finite and nonnegative"):
        LiberoInstructionNegatives(tasks, weights={**weights, 2: [1.0] * 8 + [-1.0]})
    with pytest.raises(ValueError, match="finite and nonnegative"):
        LiberoInstructionNegatives(tasks, weights={**weights, 2: [1.0] * 8 + [float("nan")]})
    with pytest.raises(ValueError, match="not all be zero"):
        LiberoInstructionNegatives(tasks, weights={**weights, 2: [0.0] * 9})
    del np


def _synthetic_prefixes(seed=0, demos=3, frames=32, dim=10):
    import numpy as np

    rng = np.random.default_rng(seed)
    base = {t: rng.normal(size=(frames, dim)) for t in range(10)}
    base[1] = base[0] + 0.02 * rng.normal(size=(frames, dim))  # task 1 starts like task 0
    prefixes = {t: [base[t] + 0.01 * rng.normal(size=(frames, dim)) for _ in range(demos)] for t in range(10)}
    prefixes[2][0] = prefixes[2][0][:20]  # ragged demonstration lengths are allowed
    return prefixes


def test_action_similarity_weights_prefer_tasks_whose_prefixes_match():
    import numpy as np

    from cosmos_framework.data.generator.action.utils.language_ranking import (
        action_prefix_distance_matrix,
        action_similarity_negative_weights,
        format_hard_negative_report,
        hard_negative_weights,
    )

    prefixes = _synthetic_prefixes()
    order, z = action_prefix_distance_matrix(prefixes)
    assert order == list(range(10)) and z.shape == (10, 10)
    assert np.allclose(z, z.T) and np.allclose(np.diag(z), 0.0)
    assert np.isclose(np.median(z[~np.eye(10, dtype=bool)]), 1.0)  # scale-free normalization
    assert z[0, 1] < 0.2 and (z[0, 2:] > 0.5).all()  # tasks 0/1 close, everything else far

    weights, order2, z2 = action_similarity_negative_weights(prefixes, fraction=1.0, temperature=0.5)
    assert order2 == order and np.allclose(z2, z)
    w0 = dict(zip([t for t in order if t != 0], weights[0]))
    # 9 candidates, z[0,1] ~ 0 and the rest ~ 1 -> p(1|0) = 1 / (1 + 8 e^-2) ~ 0.48 at temperature 0.5
    assert max(w0, key=w0.get) == 1 and w0[1] > 0.4 and abs(sum(w0.values()) - 1.0) < 1e-9
    w1 = dict(zip([t for t in order if t != 1], weights[1]))
    assert max(w1, key=w1.get) == 0
    colder = dict(zip([t for t in order if t != 0], hard_negative_weights(order, z, 1.0, 0.1)[0]))
    assert colder[1] > w0[1]  # lower temperature -> sharper
    uniform = hard_negative_weights(order, z, 0.0, 0.5)
    assert all(abs(w - 1 / 9) < 1e-12 for ws in uniform.values() for w in ws)
    mixed = dict(zip([t for t in order if t != 0], hard_negative_weights(order, z, 0.5, 0.5)[0]))
    assert all(w >= 0.5 / 9 - 1e-12 for w in mixed.values()) and mixed[1] > 0.25  # uniform floor kept

    report = format_hard_negative_report(order, z, weights, {t: f"task {t}" for t in order})
    assert "task 1" in report and "t0" in report and "->" in report

    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        hard_negative_weights(order, z, 1.5, 0.5)
    with pytest.raises(ValueError, match="positive and finite"):
        hard_negative_weights(order, z, 0.5, 0.0)
    with pytest.raises(ValueError, match="no demonstrations"):
        action_prefix_distance_matrix({**prefixes, 4: []})
    with pytest.raises(ValueError, match="one action dimension"):
        action_prefix_distance_matrix({**prefixes, 4: [np.zeros((5, 3))]})
    with pytest.raises(ValueError, match="non-empty"):
        action_prefix_distance_matrix({**prefixes, 4: [np.zeros((0, 10))]})
    # degenerate (all identical) prefixes -> zero distances, uniform weights, no division by zero
    same = {t: [np.ones((8, 10))] for t in range(10)}
    w_same, _, z_same = action_similarity_negative_weights(same, 1.0, 0.5)
    assert np.allclose(z_same, 0.0) and all(abs(w - 1 / 9) < 1e-12 for ws in w_same.values() for w in ws)
