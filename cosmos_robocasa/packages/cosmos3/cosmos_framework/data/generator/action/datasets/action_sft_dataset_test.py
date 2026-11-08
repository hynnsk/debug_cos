# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU tests of the demonstration-uniform streaming view (cosmos_hs11 v43, ``ActionEpisodeUniformIterableDataset``)."""

from __future__ import annotations

import collections

import pytest
import torch

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    ActionEpisodeUniformIterableDataset,
    ActionIterableShuffleDataset,
)


class _FakeSFT:
    """Map-style stand-in: 4 episodes with very different window counts; items are their flat index."""

    def __init__(self, lengths=(5, 50, 200, 1)):
        self.lengths = list(lengths)
        self.blocks, start = [], 0
        for n in self.lengths:
            self.blocks.append((start, n))
            start += n

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        return int(idx)

    def get_shuffle_blocks(self):
        return list(self.blocks)


def _episode_of(fake, idx):
    for e, (s, n) in enumerate(fake.blocks):
        if s <= idx < s + n:
            return e
    raise AssertionError(idx)


def test_episode_uniform_stream_equalizes_episodes_and_covers_windows():
    fake = _FakeSFT()
    ds = ActionEpisodeUniformIterableDataset(fake, seed=42)
    it = iter(ds)
    draws = [next(it) for _ in range(20000)]
    per_ep = collections.Counter(_episode_of(fake, i) for i in draws)
    # 4 episodes -> ~5000 each regardless of 5 / 50 / 200 / 1 windows
    for e in range(4):
        assert abs(per_ep[e] / 20000 - 0.25) < 0.02, per_ep
    # within an episode every window is reachable and roughly uniform
    ep2 = collections.Counter(i for i in draws if _episode_of(fake, i) == 2)
    mean = sum(ep2.values()) / 200  # ~25 draws per window: loose Poisson bounds
    assert len(ep2) == 200 and max(ep2.values()) < 2.5 * mean and min(ep2.values()) > 0.25 * mean
    # the 1-window episode always yields its only window
    assert set(i for i in draws if _episode_of(fake, i) == 3) == {255}
    # contrast: the epoch-based shuffle view yields each window once per epoch -> episode share ∝ length
    ref = ActionIterableShuffleDataset(fake, seed=0)
    first_epoch = [next(iter_ref) for iter_ref in [iter(ref)] for _ in range(len(fake))]
    assert collections.Counter(_episode_of(fake, i) for i in first_epoch) == {0: 5, 1: 50, 2: 200, 3: 1}


def test_episode_uniform_streams_are_deterministic_and_distinct_per_rank_worker():
    fake = _FakeSFT()
    a = list(zip(range(50), ActionEpisodeUniformIterableDataset.stream_indices(fake.blocks, 42, 0, 0, 2)))
    b = list(zip(range(50), ActionEpisodeUniformIterableDataset.stream_indices(fake.blocks, 42, 0, 0, 2)))
    c = list(zip(range(50), ActionEpisodeUniformIterableDataset.stream_indices(fake.blocks, 42, 1, 0, 2)))
    d = list(zip(range(50), ActionEpisodeUniformIterableDataset.stream_indices(fake.blocks, 42, 0, 1, 2)))
    assert a == b and a != c and a != d and c != d
    # shard attributes exist for RankPartitionedDataLoader and change the stream
    ds = ActionEpisodeUniformIterableDataset(fake, seed=7)
    s0 = [next(it) for it in [iter(ds)] for _ in range(20)]
    ds.shard_rank, ds.shard_world_size = 1, 2
    s1 = [next(it) for it in [iter(ds)] for _ in range(20)]
    assert s0 != s1
    with pytest.raises(ValueError, match="at least one episode"):
        next(ActionEpisodeUniformIterableDataset.stream_indices([(0, 0)], 1, 0, 0, 1))


def test_libero_factory_requires_the_streaming_loader_for_balanced_sampling(monkeypatch):
    import cosmos_framework.data.generator.action.datasets.action_sft_dataset as m

    class _DS:
        def __init__(self, **kw):
            self.kw = kw

        def __len__(self):
            return 3

        def get_shuffle_blocks(self):
            return [(0, 3)]

    class _TF:
        def __init__(self, **kw):
            pass

    monkeypatch.setattr(m, "LIBEROLeRobotDataset", _DS)
    monkeypatch.setattr(m, "ActionTransformPipeline", _TF)
    plain = m.get_action_libero_sft_dataset(root="x", iterable_shuffle=True)
    assert isinstance(plain, ActionIterableShuffleDataset)  # default: unchanged epoch-based streaming
    bal = m.get_action_libero_sft_dataset(root="x", iterable_shuffle=True, episode_balanced_sampling=True)
    assert isinstance(bal, ActionEpisodeUniformIterableDataset)
    with pytest.raises(ValueError, match="iterable_shuffle=True"):
        m.get_action_libero_sft_dataset(root="x", iterable_shuffle=False, episode_balanced_sampling=True)
