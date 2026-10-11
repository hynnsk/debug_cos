# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""LIBERO instruction negatives and paired JSON prompts for action ranking."""

from __future__ import annotations

import copy
import random


class LiberoInstructionNegatives:
    """Sample a task uniformly from the other nine tasks, then one of its captions."""

    def __init__(self, tasks: dict[int, str]):
        if len(tasks) != 10:
            raise ValueError(f"LIBERO language ranking requires exactly 10 tasks, got {len(tasks)}")
        self.captions = {key: tuple(s.strip() for s in text.split(" | ") if s.strip()) for key, text in tasks.items()}
        if any(not captions for captions in self.captions.values()):
            raise ValueError("LIBERO language ranking requires nonempty task instructions")
        self.candidates = {key: tuple(other for other in sorted(tasks) if other != key) for key in tasks}

    def sample(self, task_index: int) -> str:
        negative_task = random.choice(self.candidates[task_index])
        return random.choice(self.captions[negative_task])


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
