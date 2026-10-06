# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests for the episodic embodiment sampler and the batch packer."""

import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.action.meta.episodic_sampler import (
    EpisodicEmbodimentSampler,
    pack_samples_into_batch,
)
from cosmos_framework.data.generator.action.meta.lazy_rows import (
    EpisodeEntry,
    EpisodeTable,
    build_episode_table_from_rows,
)


def _table(num_episodes: int, num_tasks: int, frames: int, chunk: int = 16) -> EpisodeTable:
    entries, flat = [], 0
    for ep in range(num_episodes):
        n = frames + (ep % 5)  # slightly varying lengths
        entries.append(EpisodeEntry(ep, ep % num_tasks, flat, max(0, n - chunk), n))
        flat += max(0, n - chunk)
    return EpisodeTable(entries)


def test_build_episode_table_from_rows() -> None:
    episode_index = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    task_index = np.array([7] * 5 + [3] * 3 + [7] * 6)
    table = build_episode_table_from_rows(episode_index, task_index, chunk_length=2)
    assert [e.first_flat_index for e in table] == [0, 5, 8]
    assert [e.num_windows for e in table] == [3, 1, 4]
    assert [e.task_index for e in table] == [7, 3, 7]
    assert table.blocks() == [(0, 3), (5, 1), (8, 4)]
    assert table.by_task() == {7: [0, 2], 3: [1]}


def test_sampler_disjointness_and_uniform_embodiments() -> None:
    tables = {
        "many_tasks": _table(200, 40, 60),
        "two_tasks": _table(50, 2, 60),
    }
    sampler = EpisodicEmbodimentSampler(
        tables, {"many_tasks": 20, "two_tasks": 8}, k_shot=5, q_query=4, windows_per_demo=8, disjoint_tasks="auto"
    )
    assert sampler.uses_disjoint_tasks("many_tasks") and not sampler.uses_disjoint_tasks("two_tasks")
    rng = np.random.default_rng(0)
    counts = {"many_tasks": 0, "two_tasks": 0}
    for _ in range(400):
        spec = sampler.sample(rng)
        counts[spec.embodiment] += 1
        assert len(spec.support) == 5 and len(spec.query) == 4
        support_eps = {d.episode_index for d in spec.support}
        query_eps = {d.episode_index for d in spec.query}
        assert not (support_eps & query_eps)  # demonstrations never overlap
        assert len(support_eps) == 5 and len(query_eps) == 4
        table = sampler.tables[spec.embodiment]
        by_ep = {e.episode_index: e for e in table}
        for demo in spec.support + spec.query:
            e = by_ep[demo.episode_index]
            assert len(demo.window_indices) == 8
            assert all(i in e.flat_indices for i in demo.window_indices)
            assert len(set(demo.window_indices)) == 8  # enough windows -> no repeats
        if spec.embodiment == "many_tasks":
            assert spec.disjoint_tasks
            assert not ({d.task_index for d in spec.support} & {d.task_index for d in spec.query})
        else:
            assert not spec.disjoint_tasks
    # Uniform over embodiments (binomial(400, 0.5): 200 +- ~10 std).
    assert abs(counts["many_tasks"] - 200) < 50, counts


def test_sampler_short_demos_repeat_windows() -> None:
    table = EpisodeTable([EpisodeEntry(i, i % 3, i * 3, 3, 19) for i in range(12)])
    sampler = EpisodicEmbodimentSampler({"e": table}, {"e": 1}, k_shot=2, q_query=2, windows_per_demo=8)
    spec = sampler.sample(np.random.default_rng(1))
    for demo in spec.support:
        assert len(demo.window_indices) == 8
        assert set(demo.window_indices) <= set(range(demo.episode_index * 3, demo.episode_index * 3 + 3))


def _fake_sample(domain: int, caption: str = "pick") -> dict:
    return {
        "video": torch.zeros(3, 17, 32, 32, dtype=torch.uint8),
        "action": torch.zeros(16, 64),
        "action_raw": torch.zeros(16, 10),
        "text_token_ids": torch.tensor([1, 2, 3]),
        "domain_id": torch.tensor(domain),
        "raw_action_dim": torch.tensor(10),
        "conditioning_fps": torch.tensor(20),
        "image_size": torch.tensor([32.0, 32.0, 32.0, 32.0]),
        "sequence_plan": {"has_text": True},
        "ai_caption": caption,
        "mode": "wam",
        "viewpoint": "ego_view",
        "idle_frames": torch.tensor(0),
        "action_processing_record": object(),
    }


def test_pack_samples_into_batch_matches_packing_dataloader_layout() -> None:
    batch = pack_samples_into_batch([_fake_sample(7), _fake_sample(7, "place")], "fractal")
    assert batch["_num_samples"] == 2
    assert isinstance(batch["video"], list) and len(batch["video"]) == 2
    assert isinstance(batch["video"][0], list) and batch["video"][0][0].shape == (3, 17, 32, 32)
    assert isinstance(batch["action"][0], list) and batch["action"][0][0].shape == (16, 64)
    assert [int(d) for d in batch["domain_id"]] == [7, 7]
    assert len(batch["sequence_plan"]) == 2 and isinstance(batch["sequence_plan"][0], dict)
    assert len(batch["image_size"]) == 2 and batch["image_size"][0].shape == (4,)
    assert batch["conditioning_fps"][0].shape == (1,)
    assert batch["dataset_name"] == ["fractal", "fractal"]
    assert batch["raw_action_dim"][0].item() == 10


def test_pack_samples_into_batch_keeps_native_video_per_sample() -> None:
    """cosmos_hs11 v11: the native REPA-teacher frames travel like ``video`` (one [C,T,H,W] uint8 per sample)."""
    a, b = _fake_sample(7), _fake_sample(7, "place")
    a["video_native"] = torch.full((3, 17, 48, 64), 1, dtype=torch.uint8)
    b["video_native"] = torch.full((3, 17, 48, 64), 2, dtype=torch.uint8)
    batch = pack_samples_into_batch([a, b], "fractal")
    assert isinstance(batch["video_native"], list) and len(batch["video_native"]) == 2
    first = batch["video_native"][0]
    first = first[0] if isinstance(first, list) else first
    assert first.dtype == torch.uint8 and tuple(first.shape)[-4:] == (3, 17, 48, 64) and int(first.max()) == 1
    plain = pack_samples_into_batch([_fake_sample(7)], "fractal")
    assert "video_native" not in plain  # keep_native_video off -> byte-identical layout to before


def test_resize_native_video_clip_matches_the_teacher_resize() -> None:
    from cosmos_framework.data.generator.action.meta.episodic_sampler import resize_native_video_clip

    g = torch.Generator().manual_seed(0)
    clip = torch.randint(0, 256, (3, 5, 48, 64), generator=g, dtype=torch.uint8)
    out = resize_native_video_clip(clip, 16)
    assert out.shape == (3, 5, 16, 16) and out.dtype == torch.uint8 and out.is_contiguous()
    # what the DINOv2 teacher computes from the native clip (per frame, antialiased bilinear) up to uint8 rounding
    ref = torch.nn.functional.interpolate(
        clip.permute(1, 0, 2, 3).float(), size=(16, 16), mode="bilinear", align_corners=False, antialias=True
    ).permute(1, 0, 2, 3)
    assert (out.float() - ref).abs().max() <= 0.5 + 1e-4
    same = resize_native_video_clip(out, 16)
    assert same is out  # already the right size: no copy
    with pytest.raises(ValueError):
        resize_native_video_clip(clip[0], 16)


def test_synchronized_dataset_resizes_native_video_in_load() -> None:
    from cosmos_framework.data.generator.action.meta.episodic_sampler import (
        EpisodicEmbodimentSampler,
        SynchronizedMetaEpisodeIterableDataset,
    )

    class _DS:
        def __getitem__(self, i):
            s = _fake_sample(1)
            s["video_native"] = torch.zeros(3, 17, 40, 56, dtype=torch.uint8)
            return s

    sampler = EpisodicEmbodimentSampler({"a": _table(20, 5, 100)}, {"a": 1}, k_shot=2, q_query=1, windows_per_demo=2)
    d = SynchronizedMetaEpisodeIterableDataset({"a": _DS()}, sampler, max_samples_per_batch=4, seed=7, native_video_size=16)
    s = d._load("a", [0, 1])
    assert all(tuple(x["video_native"].shape) == (3, 17, 16, 16) for x in s)
    d0 = SynchronizedMetaEpisodeIterableDataset({"a": _DS()}, sampler, max_samples_per_batch=4, seed=7)
    assert tuple(d0._load("a", [0])[0]["video_native"].shape) == (3, 17, 40, 56)  # default: untouched
    dfull = SynchronizedMetaEpisodeIterableDataset(
        {"a": _DS()}, sampler, max_samples_per_batch=4, seed=7, native_video_size=16, native_video_full_res=["a"]
    )
    assert tuple(dfull._load("a", [0])[0]["video_native"].shape) == (3, 17, 40, 56)  # composite canvas: camera resolution kept


def test_synchronized_dataset_specs_identical_across_ranks_and_shards_disjoint() -> None:
    from cosmos_framework.data.generator.action.meta.episodic_sampler import (
        EpisodicEmbodimentSampler,
        SynchronizedMetaEpisodeIterableDataset,
    )

    tables = {"a": _table(20, 5, 100), "b": _table(24, 6, 100)}
    sampler = EpisodicEmbodimentSampler(tables, {"a": 1, "b": 2}, k_shot=2, q_query=2, windows_per_demo=4)
    ranks = []
    for r in range(2):
        d = SynchronizedMetaEpisodeIterableDataset({}, sampler, max_samples_per_batch=4, seed=7)
        d.shard_rank, d.shard_world_size = r, 2
        ranks.append(d)
    for j in range(6):
        s0, s1 = ranks[0].spec_for(j), ranks[1].spec_for(j)
        assert s0.embodiment == s1.embodiment and s0.support_indices == s1.support_indices and s0.query_indices == s1.query_indices
        sh0, sh1 = ranks[0].shard(s0.support_indices), ranks[1].shard(s0.support_indices)
        assert not set(sh0) & set(sh1) and sorted(sh0 + sh1) == sorted(s0.support_indices)
    # a resumed run (spec_offset) continues with different episodes
    resumed = SynchronizedMetaEpisodeIterableDataset({}, sampler, max_samples_per_batch=4, seed=7, spec_offset=100)
    assert any(resumed.spec_for(j).support_indices != ranks[0].spec_for(j).support_indices for j in range(3))
    # a materialization failure is reported for the SAME spec id instead of desynchronizing the stream
    item = next(iter(ranks[0]))  # datasets == {} -> _load raises
    assert item["failed"] is True and item["spec_id"] == 0 and item["support"] == [] and "error" in item
