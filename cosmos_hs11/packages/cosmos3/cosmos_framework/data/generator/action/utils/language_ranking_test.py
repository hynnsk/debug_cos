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
