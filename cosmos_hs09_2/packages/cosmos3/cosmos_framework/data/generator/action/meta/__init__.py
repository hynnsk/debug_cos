# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cross-embodiment few-shot (meta-learning) data + adapter utilities for Cosmos3 action heads.

Package layout (see ``docs/action_fewshot_meta.md``):

* :mod:`lazy_rows` -- column-store replacement for ``ActionBaseDataset._rows`` plus per-episode
  window tables (``EpisodeTable``) that the episodic sampler needs.
* :mod:`embodiments` -- meta-ready wrappers of the existing RT-1 / Bridge / RoboMIND readers and the
  registry that builds one ``ActionSFTDataset`` per embodiment.
* :mod:`episodic_sampler` -- ``EpisodicEmbodimentSampler`` (embodiment-uniform, demonstration-first
  support/query sampling) and ``MetaEpisodeIterableDataset`` / ``build_meta_episode_loader``.
* :mod:`meta_action_adapter` -- ``MetaActionAdapter`` holding the shared meta initialization of the
  action I/O projectors (``action2llm`` / ``llm2action`` / ``action_modality_embed``) and the
  first-order inner/outer update bookkeeping, plus ``meta_action_init.pt`` I/O.

Nothing heavy (torch, lerobot) is imported at package import time so configs stay cheap to load.
"""
