# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""LIBERO instruction negatives (uniform or action-similarity weighted) and paired JSON prompts for action ranking.

Hard negatives (cosmos_hs11 v27): the 3-demo single-scene suites fail by executing ANOTHER task's behaviour, so
the most useful negative instruction for the ranking hinge is the task whose demonstrations START like the
positive's. :func:`action_similarity_negative_weights` turns the first ``prefix_frames`` training actions of every
demonstration into a per-task mean trajectory, measures the pairwise RMS distance (standardized channels,
scale-free), and converts it into a per-task sampling distribution over the other nine tasks::

    z[i, j]        = d[i, j] / median_{i != j}(d)              (dimensionless)
    p_hard(j | i)  = softmax_{j != i}(-z[i, j] / temperature)
    p(j | i)       = (1 - fraction) / 9 + fraction * p_hard(j | i)

``fraction = 0`` (or ``negative_sampling = "uniform"``) reproduces the original uniform negatives exactly.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Mapping, Sequence

import numpy as np

NEGATIVE_SAMPLING_MODES = ("uniform", "action_similarity")


class LiberoInstructionNegatives:
    """Sample a task from the other nine tasks (uniformly, or with per-task weights), then one of its captions."""

    def __init__(self, tasks: dict[int, str], weights: Mapping[int, Sequence[float]] | None = None):
        if len(tasks) != 10:
            raise ValueError(f"LIBERO language ranking requires exactly 10 tasks, got {len(tasks)}")
        self.captions = {key: tuple(s.strip() for s in text.split(" | ") if s.strip()) for key, text in tasks.items()}
        if any(not captions for captions in self.captions.values()):
            raise ValueError("LIBERO language ranking requires nonempty task instructions")
        self.candidates = {key: tuple(other for other in sorted(tasks) if other != key) for key in tasks}
        self.weights: dict[int, list[float]] | None = None
        if weights is not None:
            if set(weights) != set(tasks):
                raise ValueError("Negative weights must be given for exactly the dataset's task ids")
            self.weights = {}
            for key, candidates in self.candidates.items():
                w = [float(x) for x in weights[key]]
                if len(w) != len(candidates):
                    raise ValueError(f"Task {key}: expected {len(candidates)} negative weights, got {len(w)}")
                if any(not math.isfinite(x) or x < 0 for x in w):
                    raise ValueError(f"Task {key}: negative weights must be finite and nonnegative")
                total = sum(w)
                if total <= 0:
                    raise ValueError(f"Task {key}: negative weights must not all be zero")
                self.weights[key] = [x / total for x in w]

    def sample(self, task_index: int) -> str:
        candidates = self.candidates[task_index]
        if self.weights is None:
            negative_task = random.choice(candidates)  # unchanged RNG consumption of the uniform recipes
        else:
            negative_task = random.choices(candidates, weights=self.weights[task_index], k=1)[0]
        return random.choice(self.captions[negative_task])


# ---------------------------------------------------------------------------------------------------------------
# Action-similarity hard negatives
# ---------------------------------------------------------------------------------------------------------------
def action_prefix_distance_matrix(prefixes: Mapping[int, Sequence[np.ndarray]]) -> tuple[list[int], np.ndarray]:
    """Scale-free RMS distance between the per-task mean action prefixes.

    ``prefixes[task]`` lists that task's demonstrations, each ``[n, D]`` = the first ``n`` training-action frames
    (``n`` may differ per demonstration). Channels are standardized over all frames of all tasks, each task's
    demonstrations are averaged per time step (a shorter demonstration stops contributing where it ends) and
    ``d[i, j]`` is the RMS difference over the time steps both tasks cover and all channels. The matrix is divided
    by its median off-diagonal entry (when positive) so a temperature on it is dimensionless. Returns the sorted
    task ids and the ``[T, T]`` matrix (zero diagonal, symmetric).
    """
    order = sorted(int(t) for t in prefixes)
    if not order:
        raise ValueError("At least one task is required")
    arrays: dict[int, list[np.ndarray]] = {}
    for task in order:
        demos = [np.asarray(a, dtype=np.float64) for a in prefixes[task]]
        if not demos:
            raise ValueError(f"Task {task} has no demonstrations; hard negatives need every task's actions")
        if any(a.ndim != 2 or a.shape[0] < 1 for a in demos):
            raise ValueError(f"Task {task}: every demonstration prefix must be a non-empty [n, D] array")
        arrays[task] = demos
    dims = {a.shape[1] for demos in arrays.values() for a in demos}
    if len(dims) != 1:
        raise ValueError(f"All demonstrations must share one action dimension, got {sorted(dims)}")
    stacked = np.concatenate([a for demos in arrays.values() for a in demos], axis=0)
    mean = stacked.mean(axis=0, keepdims=True)
    std = stacked.std(axis=0, keepdims=True) + 1e-6
    horizon = max(a.shape[0] for demos in arrays.values() for a in demos)
    dim = stacked.shape[1]
    means = np.zeros((len(order), horizon, dim))
    counts = np.zeros((len(order), horizon))
    for row, task in enumerate(order):
        for a in arrays[task]:
            n = a.shape[0]
            means[row, :n] += (a - mean) / std
            counts[row, :n] += 1.0
    means /= np.maximum(counts, 1.0)[..., None]
    distances = np.zeros((len(order), len(order)))
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            both = (counts[i] > 0) & (counts[j] > 0)
            diff = means[i, both] - means[j, both]
            distances[i, j] = distances[j, i] = float(np.sqrt(np.mean(diff**2))) if both.any() else 0.0
    off = distances[~np.eye(len(order), dtype=bool)]
    scale = float(np.median(off)) if off.size else 0.0
    if scale > 0:
        distances = distances / scale
    return order, distances


def hard_negative_weights(
    order: Sequence[int], distances: np.ndarray, fraction: float, temperature: float
) -> dict[int, list[float]]:
    """Per-task sampling weights over the OTHER tasks in ascending task-id order (the sampler's candidate order)."""
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"hard_negative_fraction must be in [0, 1], got {fraction}")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"hard_negative_temperature must be positive and finite, got {temperature}")
    distances = np.asarray(distances, dtype=np.float64)
    n = len(order)
    if distances.shape != (n, n):
        raise ValueError(f"Expected a [{n}, {n}] distance matrix, got {distances.shape}")
    if n < 2:
        raise ValueError("Hard negatives need at least two tasks")
    weights: dict[int, list[float]] = {}
    for i, task in enumerate(order):
        others = [j for j in range(n) if j != i]
        logits = -distances[i, others] / temperature
        logits -= logits.max()
        p_hard = np.exp(logits)
        p_hard /= p_hard.sum()
        mixed = (1.0 - fraction) / len(others) + fraction * p_hard
        weights[int(task)] = [float(x) for x in mixed]
    return weights


def action_similarity_negative_weights(
    prefixes: Mapping[int, Sequence[np.ndarray]], fraction: float, temperature: float
) -> tuple[dict[int, list[float]], list[int], np.ndarray]:
    """Distance matrix + mixed sampling weights in one call; see the module docstring for the formula."""
    order, distances = action_prefix_distance_matrix(prefixes)
    return hard_negative_weights(order, distances, fraction, temperature), order, distances


def format_hard_negative_report(
    order: Sequence[int],
    distances: np.ndarray,
    weights: Mapping[int, Sequence[float]],
    task_names: Mapping[int, str],
    top_k: int = 3,
) -> str:
    """Human-readable table: normalized distances and the top-k negatives (with probabilities) per task."""
    lines = ["normalized action-prefix distance z (median off-diagonal = 1):"]
    header = "      " + " ".join(f"t{t:<4d}" for t in order)
    lines.append(header)
    for i, task in enumerate(order):
        lines.append(f"t{task:<4d} " + " ".join(f"{distances[i, j]:5.2f}" for j in range(len(order))))
    lines.append("negative sampling probabilities (top-k per task):")
    for task in order:
        others = [t for t in order if t != task]
        probs = sorted(zip(weights[task], others), reverse=True)[:top_k]
        picks = ", ".join(f"t{t} {p:.2f} ({str(task_names.get(t, ''))[:30]})" for p, t in probs)
        lines.append(f"t{task:<4d} {str(task_names.get(task, ''))[:44]:44s} -> {picks}")
    return "\n".join(lines)


def tokenize_negative_instruction(prompt: dict, instruction: str, tokenizer) -> object:
    """Change only the action description; preserve every other positive-prompt field.

    The ordinary positive tokenizer retains CFG dropout. The caller masks ranking
    for dropped positives; negatives are tokenized without an independent dropout.
    """
    if not isinstance(prompt, dict) or len(prompt.get("actions", [])) != 1:
        raise ValueError("LIBERO language ranking requires a structured prompt with one action")
    negative_prompt = copy.deepcopy(prompt)
    instruction = instruction.strip()
    if not instruction.endswith((".", "!", "?")):
        instruction += "."
    negative_prompt["actions"][0]["description"] = instruction
    negative_tokenizer = copy.copy(tokenizer)  # share the processor; never instantiate a second tokenizer
    negative_tokenizer.cfg_dropout_rate = 0.0
    result = negative_tokenizer({tokenizer.input_keys[0]: negative_prompt})
    return result[tokenizer.output_keys[0]]
