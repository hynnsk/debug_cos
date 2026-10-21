"""Reptile meta-training data side: episodic (support / query) sampling over embodiments + a persistent worker stream.

Mirrors ``cosmos_hs11`` (``meta/episodic_sampler.py``): a meta-episode is one embodiment, ``k_shot`` support
demonstrations with ``windows_per_demo`` windows each, and ``q_query`` disjoint query demonstrations (diagnostics only).
Spec ``j`` is a pure function of ``(seed, j)``, so every DataLoader worker regenerates the same spec and decodes its
share of the windows (``[worker_id::num_workers]``) -- no inter-process communication, persistent workers, and the
main process reassembles episodes with :class:`EpisodeCollector` / :class:`EpisodePrefetcher`.
"""

from __future__ import annotations

import collections
import dataclasses
import itertools
import logging
import queue
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from openpi.hs11 import lerobot_v3

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SamplerConfig:
    """``action_reptile_meta_edge_v4.toml`` [custom.meta] sampling knobs."""

    k_shot: int = 32  # support demonstrations per meta-episode
    windows_per_demo: int = 8  # windows (action chunks) per support demonstration -> 256 support samples
    q_query: int = 8  # query demonstrations (zero-shot / adapted loss diagnostics)
    query_windows_per_demo: int | None = None  # defaults to windows_per_demo
    disjoint_query_tasks: bool = False  # keep support / query task sets disjoint when the embodiment has enough tasks
    min_tasks_for_disjoint: int = 4
    min_frames: int = 1  # demonstrations with fewer frames are never sampled

    @property
    def support_size(self) -> int:
        return self.k_shot * self.windows_per_demo

    @property
    def query_size(self) -> int:
        return self.q_query * (self.query_windows_per_demo or self.windows_per_demo)


@dataclasses.dataclass(frozen=True)
class MetaEpisodeSpec:
    spec_id: int
    embodiment: str
    support: tuple[tuple[int, int], ...]  # (episode_id, start frame)
    query: tuple[tuple[int, int], ...]
    disjoint_tasks: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "embodiment": self.embodiment,
            "support_episodes": sorted({e for e, _ in self.support}),
            "query_episodes": sorted({e for e, _ in self.query}),
            "num_support": len(self.support),
            "num_query": len(self.query),
            "disjoint_tasks": self.disjoint_tasks,
        }


@dataclasses.dataclass(frozen=True)
class EmbodimentTable:
    """Per-embodiment episode bookkeeping needed by the sampler (small; built in the main process)."""

    name: str
    lengths: np.ndarray  # frames per episode id
    task_keys: tuple[str, ...]  # task identity per episode id (text)

    @classmethod
    def from_dataset(cls, dataset: lerobot_v3.LeRobotV3Dataset, *, with_tasks: bool) -> "EmbodimentTable":
        lengths = np.asarray([dataset.episode_length(i) for i in range(dataset.num_episodes)], dtype=np.int64)
        tasks = tuple(dataset.episode_task_key(i) for i in range(dataset.num_episodes)) if with_tasks else tuple("" for _ in lengths)
        return cls(dataset.spec.name, lengths, tasks)


class EpisodicSampler:
    def __init__(
        self,
        tables: Mapping[str, EmbodimentTable],
        config: SamplerConfig,
        *,
        action_horizon: int,
        seed: int,
        probs: Mapping[str, float] | None = None,
    ):
        if not tables:
            raise ValueError("at least one embodiment is required")
        self.names = list(tables)
        self.tables = dict(tables)
        self.config = config
        self.action_horizon = int(action_horizon)
        self.seed = int(seed)
        p = np.asarray([probs.get(n, 1.0) if probs else 1.0 for n in self.names], dtype=np.float64)
        self.probs = p / p.sum()
        for name, table in self.tables.items():
            usable = int((table.lengths >= config.min_frames).sum())
            if usable < config.k_shot + config.q_query:
                raise ValueError(f"{name}: {usable} usable episodes < k_shot + q_query = {config.k_shot + config.q_query}")
            logger.info(f"sampler: {name}: {usable} episodes, p={self.probs[self.names.index(name)]:.3f}")

    def _windows(self, rng: np.random.Generator, length: int, count: int) -> list[int]:
        # Window starts whose chunk lies inside the episode when possible (actions are padded at the end otherwise).
        candidates = max(1, length - self.action_horizon + 1)
        if candidates >= count:
            return sorted(int(x) for x in rng.choice(candidates, size=count, replace=False))
        return sorted(int(x) for x in rng.choice(candidates, size=count, replace=True))

    def sample(self, spec_id: int, embodiment: str | None = None) -> MetaEpisodeSpec:
        rng = np.random.default_rng([self.seed, int(spec_id)])
        name = embodiment or self.names[int(rng.choice(len(self.names), p=self.probs))]
        table = self.tables[name]
        usable = np.flatnonzero(table.lengths >= self.config.min_frames)
        support_eps = rng.choice(usable, size=self.config.k_shot, replace=False)
        support_set = {int(e) for e in support_eps}
        support_tasks = {table.task_keys[e] for e in support_set}
        n_tasks = len(set(table.task_keys))
        disjoint = self.config.disjoint_query_tasks and n_tasks >= self.config.min_tasks_for_disjoint
        candidates = [int(e) for e in usable if int(e) not in support_set and (not disjoint or table.task_keys[e] not in support_tasks)]
        if len(candidates) < self.config.q_query:
            disjoint = False
            candidates = [int(e) for e in usable if int(e) not in support_set]
        query_eps = rng.choice(candidates, size=self.config.q_query, replace=False) if self.config.q_query > 0 else []
        wq = self.config.query_windows_per_demo or self.config.windows_per_demo
        support = tuple((int(e), t) for e in support_eps for t in self._windows(rng, int(table.lengths[e]), self.config.windows_per_demo))
        query = tuple((int(e), t) for e in query_eps for t in self._windows(rng, int(table.lengths[e]), wq))
        return MetaEpisodeSpec(spec_id=int(spec_id), embodiment=name, support=support, query=query, disjoint_tasks=disjoint)


# ---------------------------------------------------------------------------------------------------------------
# Worker stream
# ---------------------------------------------------------------------------------------------------------------
class ReptileStream(torch.utils.data.IterableDataset):
    """Yields ``{"spec_id", "kind", "pos", "sample"}`` items; each worker handles windows ``[worker::num_workers]``.

    ``factories[name]()`` must return ``(raw_dataset, transform)`` where ``raw_dataset`` is a ``LeRobotV3Dataset`` and
    ``transform`` maps a raw frame dict to the model inputs (built lazily inside the worker process).
    """

    def __init__(self, factories: Mapping[str, Callable[[], tuple[lerobot_v3.LeRobotV3Dataset, Callable[[dict], dict]]]], sampler: EpisodicSampler, start_spec: int = 0):
        self.factories = dict(factories)
        self.sampler = sampler
        self.start_spec = int(start_spec)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        info = torch.utils.data.get_worker_info()
        worker, num_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        loaded: dict[str, tuple[lerobot_v3.LeRobotV3Dataset, Callable[[dict], dict]]] = {}
        for spec_id in itertools.count(self.start_spec):
            spec = self.sampler.sample(spec_id)
            items = [("support", i, e, t) for i, (e, t) in enumerate(spec.support)]
            items += [("query", i, e, t) for i, (e, t) in enumerate(spec.query)]
            if spec.embodiment not in loaded:
                loaded[spec.embodiment] = self.factories[spec.embodiment]()
            dataset, transform = loaded[spec.embodiment]
            for kind, pos, episode_id, t in items[worker::num_workers]:
                try:
                    sample = transform(dataset.get_frame(episode_id, t))
                except Exception as e:  # noqa: BLE001 -- a broken video must not kill the run
                    logger.warning(f"[{spec.embodiment}] spec {spec_id} episode {episode_id} t={t} failed ({type(e).__name__}: {e}); marking failed")
                    sample = None
                yield {"spec_id": spec_id, "kind": kind, "pos": pos, "sample": sample}

    def __len__(self) -> int:  # unused (infinite stream); required by some DataLoader checks
        return 10**9


@dataclasses.dataclass
class MetaEpisodeData:
    spec: MetaEpisodeSpec
    support: list[dict[str, Any]]
    query: list[dict[str, Any]]
    failed: bool = False


class EpisodeCollector:
    """Reorders the worker stream into complete meta-episodes (spec ids arrive interleaved across workers)."""

    def __init__(self, items: Iterator[dict[str, Any]], sampler: EpisodicSampler, start_spec: int = 0):
        self._items = items
        self._sampler = sampler
        self._next = int(start_spec)
        self._buffers: dict[int, dict[str, list]] = collections.defaultdict(lambda: {"support": [], "query": []})
        self._specs: dict[int, MetaEpisodeSpec] = {}

    def _spec(self, spec_id: int) -> MetaEpisodeSpec:
        if spec_id not in self._specs:
            self._specs[spec_id] = self._sampler.sample(spec_id)
        return self._specs[spec_id]

    def _complete(self, spec_id: int) -> bool:
        spec = self._spec(spec_id)
        buf = self._buffers[spec_id]
        return len(buf["support"]) == len(spec.support) and len(buf["query"]) == len(spec.query)

    def __iter__(self) -> Iterator[MetaEpisodeData]:
        while True:
            while not self._complete(self._next):
                item = next(self._items)
                sid = int(item["spec_id"])
                if sid < self._next:
                    continue  # stale item from a previous (already emitted) spec
                self._buffers[sid][item["kind"]].append((int(item["pos"]), item["sample"]))
            spec = self._spec(self._next)
            buf = self._buffers.pop(self._next)
            support = [s for _, s in sorted(buf["support"], key=lambda x: x[0])]
            query = [s for _, s in sorted(buf["query"], key=lambda x: x[0])]
            failed = any(s is None for s in support) or any(s is None for s in query)
            self._specs.pop(self._next, None)
            self._next += 1
            yield MetaEpisodeData(spec=spec, support=support, query=query, failed=failed)


class EpisodePrefetcher:
    """Background thread that keeps up to ``maxsize`` assembled meta-episodes ready while the GPU trains."""

    def __init__(self, collector: EpisodeCollector, maxsize: int = 2):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._collector = collector
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="reptile-prefetch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for episode in self._collector:
                if self._stop.is_set():
                    return
                self._queue.put(episode)
        except Exception as e:  # noqa: BLE001
            logger.exception("prefetch thread died")
            self._queue.put(e)

    def get(self) -> MetaEpisodeData:
        item = self._queue.get()
        if isinstance(item, Exception):
            raise item
        return item

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------------------------------------------
_BATCH_KEYS = ("image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask", "actions")


def collate(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack transformed samples into one batch dict (only the model input keys)."""
    import jax

    picked = [{k: s[k] for k in _BATCH_KEYS if k in s} for s in samples]
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *picked)


def micro_batches(samples: Sequence[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    """Split a support / query set into equally sized micro-batches (the set size must be a multiple)."""
    n = len(samples)
    if n % batch_size != 0:
        raise ValueError(f"set size {n} is not a multiple of the micro-batch size {batch_size}")
    return [collate(samples[i : i + batch_size]) for i in range(0, n, batch_size)]


def meta_lr_at(iteration: int, max_iter: int, meta_lr: float, min_ratio: float) -> float:
    """Linear decay ``meta_lr -> meta_lr * min_ratio`` over ``max_iter`` meta-iterations (cosmos_hs11 outer schedule)."""
    if max_iter <= 1:
        return meta_lr
    frac = min(max(iteration / float(max_iter), 0.0), 1.0)
    return meta_lr * (1.0 - (1.0 - min_ratio) * frac)
