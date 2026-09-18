# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Episodic (support / query) sampling over embodiments for few-shot meta-training.

A *meta-episode* is one embodiment plus two disjoint sets of demonstrations::

    embodiment e ~ Uniform({fractal, bridge, robomind_ur, robomind_franka, molmoact2_yam})
    support demos  S = K episodes of e            (K-shot == K demonstrations, not K windows)
    query demos    Q = Q episodes of e, S ∩ Q = ∅ (and task(S) ∩ task(Q) = ∅ when e has enough tasks)
    windows        w windows sampled inside every chosen demonstration

Sampling embodiments uniformly (not proportionally to their size) is deliberate: RT-1 has 87K
episodes and the YAM subset a few thousand, so a size-proportional mixture would be dominated by
the Google Robot. Sampling *demonstrations first* is what makes ``K`` a demonstration count -- the
quantity the downstream LIBERO few-shot protocol is expressed in ("3 demos per task").

:class:`EpisodicEmbodimentSampler` is pure bookkeeping over ``EpisodeTable``s (no I/O, seedable).
:class:`MetaEpisodeIterableDataset` materializes a meta-episode inside a DataLoader worker: it
pulls the sampled windows through the per-embodiment ``ActionSFTDataset`` (video decode + the
``ActionTransformPipeline``) and packs them into the ``PackingDataLoader``-style batches that
``OmniMoTModel.training_step`` consumes (see :func:`pack_samples_into_batch`).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from cosmos_framework.data.generator.action.meta.lazy_rows import EpisodeTable
from cosmos_framework.utils import log

__all__ = [
    "EpisodicEmbodimentSampler",
    "MetaEpisodeIterableDataset",
    "MetaEpisodeLoader",
    "MetaEpisodeSpec",
    "SampledDemo",
    "build_meta_episode_loader",
    "pack_samples_into_batch",
]


# --------------------------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SampledDemo:
    """One demonstration and the flat dataset indices of the windows drawn from it."""

    episode_index: int
    task_index: int
    window_indices: tuple[int, ...]


@dataclass
class MetaEpisodeSpec:
    """Everything needed to materialize one meta-episode (embodiment + support/query windows)."""

    embodiment: str
    domain_id: int
    support: list[SampledDemo]
    query: list[SampledDemo]
    disjoint_tasks: bool  # whether support/query task sets were kept disjoint for this episode

    @property
    def support_indices(self) -> list[int]:
        return [i for d in self.support for i in d.window_indices]

    @property
    def query_indices(self) -> list[int]:
        return [i for d in self.query for i in d.window_indices]

    def summary(self) -> dict[str, Any]:
        return {
            "embodiment": self.embodiment,
            "domain_id": self.domain_id,
            "num_support_demos": len(self.support),
            "num_query_demos": len(self.query),
            "num_support_windows": len(self.support_indices),
            "num_query_windows": len(self.query_indices),
            "support_tasks": sorted({d.task_index for d in self.support}),
            "query_tasks": sorted({d.task_index for d in self.query}),
            "disjoint_tasks": self.disjoint_tasks,
        }


# --------------------------------------------------------------------------------------------
# Sampler (pure bookkeeping)
# --------------------------------------------------------------------------------------------
class EpisodicEmbodimentSampler:
    """Embodiment-uniform, demonstration-first support/query sampler.

    Args:
        episode_tables: ``{embodiment_name: EpisodeTable}``.
        domain_ids: ``{embodiment_name: domain_id}`` (only carried into the spec).
        k_shot: Number of support demonstrations.
        q_query: Number of query demonstrations.
        windows_per_demo: Windows drawn from each support demonstration.
        query_windows_per_demo: Windows per query demonstration (defaults to ``windows_per_demo``).
        disjoint_tasks: ``"auto"`` keeps support and query task sets disjoint whenever the
            embodiment has at least ``min_tasks_for_disjoint`` tasks (RoboMIND-Franka has 2 tasks and
            therefore falls back to *episode*-disjoint sampling); ``True``/``False`` force it.
        min_tasks_for_disjoint: See ``disjoint_tasks``.
        min_windows_per_demo: Demonstrations with fewer valid windows are never sampled.
        embodiment_weights: Optional ``{name: weight}``; ``None`` == uniform ``p(e) = 1/E``.
    """

    def __init__(
        self,
        episode_tables: dict[str, EpisodeTable],
        domain_ids: dict[str, int],
        k_shot: int = 5,
        q_query: int = 5,
        windows_per_demo: int = 8,
        query_windows_per_demo: int | None = None,
        disjoint_tasks: bool | str = "auto",
        min_tasks_for_disjoint: int = 4,
        min_windows_per_demo: int = 1,
        embodiment_weights: dict[str, float] | None = None,
    ) -> None:
        if k_shot < 1 or q_query < 1:
            raise ValueError(f"k_shot and q_query must be >= 1, got {k_shot}, {q_query}")
        if windows_per_demo < 1:
            raise ValueError("windows_per_demo must be >= 1")
        self.k_shot = int(k_shot)
        self.q_query = int(q_query)
        self.windows_per_demo = int(windows_per_demo)
        self.query_windows_per_demo = int(query_windows_per_demo or windows_per_demo)
        self.min_windows_per_demo = max(1, int(min_windows_per_demo))
        self.disjoint_tasks_mode = disjoint_tasks
        self.min_tasks_for_disjoint = int(min_tasks_for_disjoint)
        self.domain_ids = dict(domain_ids)

        self.names: list[str] = []
        self.tables: dict[str, EpisodeTable] = {}
        self._by_task: dict[str, dict[int, list[int]]] = {}
        for name, table in episode_tables.items():
            usable = table.usable(self.min_windows_per_demo)
            if len(usable) < self.k_shot + self.q_query:
                raise ValueError(
                    f"Embodiment {name!r} has only {len(usable)} usable demonstrations "
                    f"(need k_shot + q_query = {self.k_shot + self.q_query})."
                )
            self.names.append(name)
            self.tables[name] = usable
            self._by_task[name] = usable.by_task()
        if not self.names:
            raise ValueError("No embodiments given.")

        weights = np.ones(len(self.names), dtype=np.float64)
        if embodiment_weights:
            weights = np.asarray([float(embodiment_weights.get(n, 0.0)) for n in self.names], dtype=np.float64)
            if weights.sum() <= 0:
                raise ValueError("embodiment_weights must have a positive total.")
        self.probs = weights / weights.sum()
        log.info(
            "EpisodicEmbodimentSampler: "
            + ", ".join(
                f"{n}: {len(self.tables[n])} demos / {len(self._by_task[n])} tasks / p={p:.3f}"
                for n, p in zip(self.names, self.probs)
            )
        )

    # ---- helpers -------------------------------------------------------------
    def uses_disjoint_tasks(self, name: str) -> bool:
        if isinstance(self.disjoint_tasks_mode, bool):
            return self.disjoint_tasks_mode
        return len(self._by_task[name]) >= self.min_tasks_for_disjoint

    def sample_embodiment(self, rng: np.random.Generator) -> str:
        return self.names[int(rng.choice(len(self.names), p=self.probs))]

    @staticmethod
    def _sample_windows(rng: np.random.Generator, first: int, num: int, count: int) -> tuple[int, ...]:
        if count >= num:
            # Every window once, in random order, then repeat to reach ``count``.
            picks = rng.permutation(num)
            reps = int(math.ceil(count / num))
            picks = np.concatenate([picks] + [rng.permutation(num) for _ in range(reps - 1)])[:count]
        else:
            picks = rng.choice(num, size=count, replace=False)
        return tuple(int(first + p) for p in picks)

    def _demos(self, rng: np.random.Generator, name: str, positions: Sequence[int], count: int) -> list[SampledDemo]:
        table = self.tables[name]
        out = []
        for pos in positions:
            e = table[int(pos)]
            out.append(
                SampledDemo(
                    episode_index=e.episode_index,
                    task_index=e.task_index,
                    window_indices=self._sample_windows(rng, e.first_flat_index, e.num_windows, count),
                )
            )
        return out

    # ---- main entry --------------------------------------------------------------
    def sample(self, rng: np.random.Generator, embodiment: str | None = None) -> MetaEpisodeSpec:
        name = embodiment or self.sample_embodiment(rng)
        table = self.tables[name]
        n = len(table)
        support_pos = rng.choice(n, size=self.k_shot, replace=False)
        support_tasks = {table[int(p)].task_index for p in support_pos}
        support_set = set(int(p) for p in support_pos)

        disjoint = self.uses_disjoint_tasks(name)
        candidates: list[int] = []
        if disjoint:
            candidates = [
                p for t, ps in self._by_task[name].items() if t not in support_tasks for p in ps if p not in support_set
            ]
            if len(candidates) < self.q_query:
                disjoint = False
        if not disjoint:
            candidates = [p for p in range(n) if p not in support_set]
        query_pos = rng.choice(len(candidates), size=self.q_query, replace=False)
        query_pos = [candidates[int(p)] for p in query_pos]

        return MetaEpisodeSpec(
            embodiment=name,
            domain_id=self.domain_ids[name],
            support=self._demos(rng, name, [int(p) for p in support_pos], self.windows_per_demo),
            query=self._demos(rng, name, query_pos, self.query_windows_per_demo),
            disjoint_tasks=disjoint,
        )


# --------------------------------------------------------------------------------------------
# Packing samples into the model's batch format
# --------------------------------------------------------------------------------------------
def pack_samples_into_batch(samples: list[dict[str, Any]], dataset_name: str) -> dict[str, Any]:
    """Pack transformed samples the way ``PackingDataLoader`` does (list-of-per-sample values).

    Reuses ``custom_collate_fn`` and the ``JointDataLoader`` key conventions so the result is
    byte-for-byte the structure ``OmniMoTModel.training_step`` expects: multi-item keys
    (``video``/``action``/``text_token_ids``/...) become ``list[list[Tensor]]``, per-sequence metadata
    (``sequence_plan``/``domain_id``/...) become flat lists, tensor-origin keys ``list[Tensor(1,...)]``.
    """
    from cosmos_framework.data.generator.joint_dataloader import (
        _BATCH_TIMING_KEYS,
        JointDataLoader,
        custom_collate_fn,
    )

    if not samples:
        raise ValueError("pack_samples_into_batch needs at least one sample")
    multi_item_keys = JointDataLoader._MULTI_ITEM_KEYS
    flatten_keys = JointDataLoader._FLATTEN_LIST_KEYS
    output: dict[str, Any] = {}
    for sample in samples:
        collated = custom_collate_fn([sample])
        split: dict[str, Any] = {}
        for k, v in collated.items():
            if k in _BATCH_TIMING_KEYS:
                split[k] = v
            elif isinstance(v, list) and k in multi_item_keys:
                elem = v[0]
                split[k] = elem if isinstance(elem, list) else v[0:1]
            elif isinstance(v, list):
                split[k] = v[0]
            elif isinstance(v, torch.Tensor):
                split[k] = v[0:1]
            else:
                split[k] = v
        split["dataset_name"] = dataset_name
        for k, v in split.items():
            if k in _BATCH_TIMING_KEYS:
                output.setdefault(k, v)
            elif k in flatten_keys and isinstance(v, list):
                output.setdefault(k, []).extend(v)
            else:
                output.setdefault(k, []).append(v)
    output["_num_samples"] = len(samples)
    return output


def _chunk(indices: list[int], size: int) -> list[list[int]]:
    size = max(1, int(size))
    return [indices[i : i + size] for i in range(0, len(indices), size)]


# --------------------------------------------------------------------------------------------
# Iterable dataset + loader
# --------------------------------------------------------------------------------------------
class MetaEpisodeIterableDataset(IterableDataset):
    """Infinite stream of materialized meta-episodes (one dict per meta-episode).

    Each yielded item::

        {
          "embodiment": str, "domain_id": int, "spec": dict (MetaEpisodeSpec.summary()),
          "support": [packed_batch, ...],   # len == ceil(K * w / max_samples_per_batch)
          "query":   [packed_batch, ...],
        }

    ``shard_rank`` / ``shard_world_size`` (set by :class:`MetaEpisodeLoader`) and the DataLoader
    worker id are folded into the RNG seed so every (rank, worker) draws different episodes; the
    stream is reproducible for a fixed ``seed``.
    """

    def __init__(
        self,
        datasets: dict[str, Any],
        sampler: EpisodicEmbodimentSampler,
        max_samples_per_batch: int = 40,
        seed: int = 42,
        embodiment_override: str | None = None,
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.sampler = sampler
        self.max_samples_per_batch = int(max_samples_per_batch)
        self.seed = int(seed)
        self.embodiment_override = embodiment_override
        self.max_consecutive_failures = 8  # a systematic error (missing stats, bad root) must surface, not spin
        self.shard_rank = 0
        self.shard_world_size = 1

    def _load(self, name: str, indices: list[int]) -> list[dict[str, Any]]:
        dataset = self.datasets[name]
        return [dataset[int(i)] for i in indices]

    def materialize(self, spec: MetaEpisodeSpec) -> dict[str, Any]:
        support = [
            pack_samples_into_batch(self._load(spec.embodiment, chunk), spec.embodiment)
            for chunk in _chunk(spec.support_indices, self.max_samples_per_batch)
        ]
        query = [
            pack_samples_into_batch(self._load(spec.embodiment, chunk), spec.embodiment)
            for chunk in _chunk(spec.query_indices, self.max_samples_per_batch)
        ]
        return {
            "embodiment": spec.embodiment,
            "domain_id": spec.domain_id,
            "spec": spec.summary(),
            "support": support,
            "query": query,
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        from cosmos_framework.data.generator.action.utils.video_decode_env import ensure_video_decoder_libs

        ensure_video_decoder_libs()
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1
        global_shard = int(self.shard_rank) * num_workers + worker_id
        rng = np.random.default_rng([self.seed, int(self.shard_world_size), global_shard])
        consecutive_failures = 0
        while True:
            spec = self.sampler.sample(rng, embodiment=self.embodiment_override)
            try:
                item = self.materialize(spec)
            except Exception as e:  # noqa: BLE001 - one bad episode must not kill the worker
                consecutive_failures += 1
                log.warning(
                    f"meta-episode materialization failed ({spec.embodiment}): {type(e).__name__}: {e}; "
                    f"resampling ({consecutive_failures}/{self.max_consecutive_failures})"
                )
                if consecutive_failures >= self.max_consecutive_failures:
                    raise RuntimeError(
                        f"{consecutive_failures} consecutive meta-episode failures; last error: {type(e).__name__}: {e}"
                    ) from e
                continue
            consecutive_failures = 0
            yield item


class MetaEpisodeLoader:
    """Rank-aware wrapper around ``DataLoader(MetaEpisodeIterableDataset, batch_size=None)``."""

    def __init__(
        self,
        dataset: MetaEpisodeIterableDataset,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        timeout_s: float = 1800.0,
    ) -> None:
        rank, world_size = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        dataset.shard_rank = rank
        dataset.shard_world_size = world_size
        self.dataset = dataset
        kwargs: dict[str, Any] = dict(batch_size=None, num_workers=int(num_workers), pin_memory=pin_memory)
        if num_workers > 0:
            # ``timeout`` turns a lost/stalled worker item into a RuntimeError instead of an infinite
            # wait on next(loader) (which deadlocks the other ranks in all_reduce).
            kwargs.update(prefetch_factor=int(prefetch_factor), persistent_workers=True, timeout=float(timeout_s))
        self.dataloader = DataLoader(dataset, **kwargs)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.dataloader)


@dataclass
class MetaSamplerConfig:
    """Sampler knobs, kept as a dataclass so ``[custom.meta]`` TOML overrides map 1:1."""

    k_shot: int = 5
    q_query: int = 5
    windows_per_demo: int = 8
    query_windows_per_demo: int | None = None
    disjoint_tasks: bool | str = "auto"
    min_tasks_for_disjoint: int = 4
    min_windows_per_demo: int = 1
    embodiment_weights: dict[str, float] | None = None
    max_samples_per_batch: int = 40
    num_workers: int = 4
    prefetch_factor: int = 2
    loader_timeout_s: float = 1800.0
    seed: int = 42
    embodiment_override: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def build_meta_episode_loader(
    *,
    data_root: str,
    embodiments: Sequence[str],
    tokenizer_config: dict | None,
    max_action_dim: int = 64,
    chunk_length: int = 16,
    mode: str = "wam",
    resolution: str | int | None = "256",
    cfg_dropout_rate: float = 0.1,
    format_prompt_as_json: bool = True,
    append_idle_frames: bool = True,
    dataset_kwargs: dict[str, dict[str, Any]] | None = None,
    root_overrides: dict[str, str] | None = None,
    k_shot: int = 5,
    q_query: int = 5,
    windows_per_demo: int = 8,
    query_windows_per_demo: int | None = None,
    disjoint_tasks: bool | str = "auto",
    min_tasks_for_disjoint: int = 4,
    min_windows_per_demo: int = 1,
    embodiment_weights: dict[str, float] | None = None,
    max_samples_per_batch: int = 40,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    loader_timeout_s: float = 1800.0,
    seed: int = 42,
    embodiment_override: str | None = None,
) -> MetaEpisodeLoader:
    """Build every embodiment dataset, the episodic sampler, and the rank-aware loader.

    This is the ``dataloader_train`` entry point of the ``action_fewshot_meta_edge`` experiment
    (``LazyCall(build_meta_episode_loader)(...)``); ``[custom.meta]`` TOML keys with the same names
    override the sampler arguments (see ``scripts/train_action_meta.py``).
    """
    from cosmos_framework.data.generator.action.meta.embodiments import (
        build_embodiment_sft_dataset,
        build_meta_transform,
        get_episode_table,
    )

    transform = build_meta_transform(
        tokenizer_config=tokenizer_config,
        max_action_dim=max_action_dim,
        cfg_dropout_rate=cfg_dropout_rate,
        append_idle_frames=append_idle_frames,
        format_prompt_as_json=format_prompt_as_json,
    )
    datasets: dict[str, Any] = {}
    tables: dict[str, EpisodeTable] = {}
    domain_ids: dict[str, int] = {}
    for name in embodiments:
        ds = build_embodiment_sft_dataset(
            name,
            data_root,
            transform,
            resolution=resolution,
            chunk_length=chunk_length,
            mode=mode,
            root_override=(root_overrides or {}).get(name),
            dataset_kwargs=(dataset_kwargs or {}).get(name),
        )
        datasets[name] = ds
        tables[name] = get_episode_table(ds)
        domain_ids[name] = int(ds._dataset.domain_id)
    sampler = EpisodicEmbodimentSampler(
        tables,
        domain_ids,
        k_shot=k_shot,
        q_query=q_query,
        windows_per_demo=windows_per_demo,
        query_windows_per_demo=query_windows_per_demo,
        disjoint_tasks=disjoint_tasks,
        min_tasks_for_disjoint=min_tasks_for_disjoint,
        min_windows_per_demo=min_windows_per_demo,
        embodiment_weights=embodiment_weights,
    )
    stream = MetaEpisodeIterableDataset(
        datasets,
        sampler,
        max_samples_per_batch=max_samples_per_batch,
        seed=seed,
        embodiment_override=embodiment_override,
    )
    return MetaEpisodeLoader(
        stream, num_workers=num_workers, prefetch_factor=prefetch_factor, timeout_s=loader_timeout_s
    )
