# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Network-side glue of the REPA loss: intermediate MoT video tokens -> projector, teacher tokens -> target.

Lives inside ``Cosmos3VFMNetwork`` (as ``repa_head``) so every learnable REPA parameter is part of the
FSDP root unit, the EMA copy and the DCP checkpoint. It is called from ``Cosmos3VFMNetwork.forward`` with

* ``vision_hidden``: the hidden state of *all* vision tokens at the selected decoder layer, in packing order
  (sample-major, then ``t, h, w`` -- see ``patchify_and_pack_latents`` / ``pack_vision_tokens``);
* ``token_shapes``: per-sample ``(T, H, W)`` MoT token grids;
* ``noisy_frame_indexes``: per-sample latent frames that are denoised (the clean conditioning frame 0 is
  never aligned);
* ``teacher_tokens``: per-sample V-JEPA tokens ``[V, T_t, H_p, W_p, D_t]`` (one clip per camera view,
  covering the raw frames behind latent frames ``1..T-1``).

For each sample the teacher grid of each view is adapted to ``(T-1, H, W/V)``, the views are concatenated
along width (the ``concat_view`` canvas layout), the noisy frames are selected and everything is flattened
to ``[N, D_t]`` so the loss is a single cosine over tokens.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from cosmos_framework.model.generator.repa.adapters import (
    build_projector,
    build_target_adapter,
    concat_views_along_width_subgrid,
    split_flat_tokens,
)

Grid3 = tuple[int, int, int]


class RepaTeacherTokens:
    """Single-use holder for the raw teacher tokens handed to ``Cosmos3VFMNetwork.forward``.

    The network takes the tokens out (``take()``) as soon as it has built the small adapted targets, so the
    ~0.8 GB raw tensor is not kept alive by the caller's argument list through the MoT forward/backward.
    """

    def __init__(self, tokens: torch.Tensor | Sequence[torch.Tensor]) -> None:
        self._tokens: torch.Tensor | Sequence[torch.Tensor] | None = tokens

    def take(self) -> torch.Tensor | Sequence[torch.Tensor]:
        if self._tokens is None:
            raise RuntimeError("REPA teacher tokens were already consumed")
        tokens, self._tokens = self._tokens, None
        return tokens

    @property
    def consumed(self) -> bool:
        return self._tokens is None


class RepaAlignmentHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        teacher_embed_dim: int,
        projector_hidden_dim: int = 2048,
        projector_type: str = "mlp",
        target_adapter: str = "avgpool",
        adapter_kernel_size: int = 3,
        adapter_depthwise: bool = True,
        teacher_grid_thw: Grid3 = (8, 16, 16),
        target_grid_thw: Grid3 = (4, 5, 5),
        num_views: int = 2,
        target_subgrid_thw: Grid3 = (1, 1, 1),
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.teacher_embed_dim = int(teacher_embed_dim)
        self.num_views = int(num_views)
        self.teacher_grid_thw = tuple(int(v) for v in teacher_grid_thw)
        self.target_grid_thw = tuple(int(v) for v in target_grid_thw)
        # "Less pooling" (cosmos_hs10 v13/v14, hs11 v5): teacher cells predicted per MoT token along (t, h, w). The
        # teacher is adapted to the (target_grid * subgrid) grid, the S = st*sh*sw cells of a token become consecutive
        # rows ([N*S, D_t]) and the projector emits all of them at once (S * D_t wide).
        self.target_subgrid_thw = tuple(int(v) for v in target_subgrid_thw)
        if len(self.target_subgrid_thw) != 3 or min(self.target_subgrid_thw) < 1:
            raise ValueError(f"target_subgrid_thw must be three positive ints, got {target_subgrid_thw}")
        for name, tg, sg in zip("thw", self.teacher_grid_thw, self._refine(self.target_grid_thw)):
            if sg > tg:
                raise ValueError(
                    f"target_grid_thw x target_subgrid_thw = {self._refine(self.target_grid_thw)} exceeds the teacher grid "
                    f"{self.teacher_grid_thw} along {name}; a sub-cell cannot be finer than one teacher token."
                )
        self.num_subcells = self.target_subgrid_thw[0] * self.target_subgrid_thw[1] * self.target_subgrid_thw[2]
        self.target_adapter_name = target_adapter
        self.projector_type = projector_type
        # "mlp": REPA's Linear-SiLU-Linear-SiLU-Linear (hidden ``projector_hidden_dim``); "linear": one Linear.
        self.projector = build_projector(
            projector_type, self.hidden_size, int(projector_hidden_dim), self.num_subcells * self.teacher_embed_dim
        )
        self.target_adapter = build_target_adapter(
            target_adapter,
            self.teacher_embed_dim,
            kernel_size=adapter_kernel_size,
            in_grid=self.teacher_grid_thw,  # type: ignore[arg-type]
            out_grid=self._refine(self.target_grid_thw),  # type: ignore[arg-type]
            depthwise=adapter_depthwise,
        )

    def _refine(self, grid: Grid3) -> Grid3:
        """Per-view MoT grid -> per-view adapted teacher grid (``grid * target_subgrid_thw``)."""
        return tuple(int(g) * int(s) for g, s in zip(grid, self.target_subgrid_thw))  # type: ignore[return-value]

    def reset_parameters(self) -> None:
        self.projector.reset_parameters()
        self.target_adapter.reset_parameters()

    # ------------------------------------------------------------------------------------------
    def probe(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Zero-weighted pass through every REPA parameter (keeps them in the autograd graph on ranks/steps
        without alignable tokens, which FSDP's gradient reduction requires)."""
        z = torch.zeros(1, self.hidden_size, device=device, dtype=dtype)
        p = self.projector(z).sum()
        t = torch.zeros(1, self.teacher_embed_dim, *self.teacher_grid_thw, device=device, dtype=dtype)
        a = self.target_adapter(t, self._refine(self.target_grid_thw)).sum()
        return 0.0 * (p + a)

    def adapt_targets(self, teacher_tokens: torch.Tensor, out_grid: Grid3) -> torch.Tensor:
        """``[B*V, T_t, H_p, W_p, D_t]`` -> ``[B, T, H, V*Wv, S, D_t]`` with ``out_grid = (T, H, Wv)`` the per-view MoT
        grid and ``S = prod(target_subgrid_thw)`` teacher sub-cells per MoT token (``S = 1`` by default)."""
        x = teacher_tokens.permute(0, 4, 1, 2, 3)  # [B*V, D_t, T_t, H_p, W_p]
        y = self.target_adapter(x, self._refine(out_grid))  # [B*V, D_t, T*st, H*sh, Wv*sw]
        return concat_views_along_width_subgrid(y, self.num_views, self.target_subgrid_thw)  # [B,T,H,V*Wv,S,D_t]

    # ------------------------------------------------------------------------------------------
    def _prepare_noisy_indexes(
        self, noisy_frame_indexes: Sequence[torch.Tensor], device: torch.device
    ) -> list[torch.Tensor]:
        """Move the per-sample noisy latent-frame indexes to ``device`` and reject alignment of latent frame 0
        (the clean conditioning frame has no teacher target). One device->host check for the whole batch."""
        nfi_list = [nfi.to(device=device, dtype=torch.long) for nfi in noisy_frame_indexes]
        non_empty = [nfi for nfi in nfi_list if nfi.numel() > 0]
        if non_empty and bool((torch.cat(non_empty) < 1).any()):
            bad = [i for i, nfi in enumerate(nfi_list) if nfi.numel() > 0 and bool((nfi < 1).any())]
            raise ValueError(
                "REPA aligns the predicted (noisy) latent frames only; latent frame 0 is the clean conditioning "
                f"frame and has no teacher target, but samples {bad} noise frame 0."
            )
        return nfi_list

    def compute_targets(
        self,
        teacher_tokens: torch.Tensor | Sequence[torch.Tensor],
        token_shapes: Sequence[Sequence[int]],
        noisy_frame_indexes: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, list[int]] | None:
        """Adapt the teacher tokens onto the MoT token grids and select the predicted frames.

        ``teacher_tokens``: one ``[V,T_t,H_p,W_p,D_t]`` tensor per sample, or a single batched
        ``[B,V,T_t,H_p,W_p,D_t]`` tensor (all samples share the teacher grid).

        Returns ``(target [N,D_t], tokens per sample)`` or ``None`` when no sample has a predicted frame. Meant to run
        BEFORE the MoT forward so the raw teacher tokens (~0.8 GB for 128 windows x 2 views) can be released early;
        the result is ~40 MB.
        """
        batched_teacher = isinstance(teacher_tokens, torch.Tensor)
        num_teacher = teacher_tokens.shape[0] if batched_teacher else len(teacher_tokens)
        if len(token_shapes) != num_teacher or len(token_shapes) != len(noisy_frame_indexes):
            raise ValueError(
                f"Got {len(token_shapes)} vision samples, {num_teacher} teacher clips and "
                f"{len(noisy_frame_indexes)} noisy-frame index sets; they must match one-to-one."
            )
        if batched_teacher and teacher_tokens.ndim != 6:
            raise ValueError(f"Batched teacher tokens must be [B,V,T_t,H_p,W_p,D_t], got {tuple(teacher_tokens.shape)}")
        device = teacher_tokens.device if batched_teacher else teacher_tokens[0].device
        nfi_list = self._prepare_noisy_indexes(noisy_frame_indexes, device)

        # Fast path: every sample shares the MoT grid and the teacher grid (the LIBERO case) -> one adapter call.
        uniform = len({tuple(int(v) for v in s) for s in token_shapes}) == 1 and (
            batched_teacher or len({tuple(t.shape) for t in teacher_tokens}) == 1
        )
        adapted_all: torch.Tensor | None = None
        if uniform:
            t_len, h_tok, w_tok = (int(v) for v in token_shapes[0])
            out_grid = self._per_view_grid(t_len, h_tok, w_tok)
            if batched_teacher:
                stacked = teacher_tokens.flatten(0, 1)  # [B*V, T_t, H_p, W_p, D_t] (view-major)
            else:
                stacked = torch.cat(list(teacher_tokens), dim=0)  # [B*V, T_t, H_p, W_p, D_t]
            adapted_all = self.adapt_targets(stacked, out_grid)  # [B, T-1, H, W, S, D_t]

        targets: list[torch.Tensor] = []
        counts: list[int] = []
        for i, (shape, nfi) in enumerate(zip(token_shapes, nfi_list)):
            t_len, h_tok, w_tok = (int(v) for v in shape)
            if nfi.numel() == 0:
                counts.append(0)
                continue
            if adapted_all is not None:
                target_grid = adapted_all[i]  # [T-1, H, W, S, D_t]
            else:
                tt = teacher_tokens[i]
                if tt.ndim != 5:
                    raise ValueError(f"teacher_tokens[{i}] must be [V,T_t,H_p,W_p,D_t], got {tuple(tt.shape)}")
                target_grid = self.adapt_targets(tt, self._per_view_grid(t_len, h_tok, w_tok))[0]
            # rows: frame-major, then h, w, then the S sub-cells of that token -> [n_i * S, D_t]
            targets.append(target_grid.index_select(0, nfi - 1).reshape(-1, target_grid.shape[-1]))
            counts.append(int(targets[-1].shape[0]))
        if not targets:
            return None
        return torch.cat(targets, dim=0), counts

    def project(
        self,
        vision_hidden: torch.Tensor,
        token_shapes: Sequence[Sequence[int]],
        noisy_frame_indexes: Sequence[torch.Tensor],
    ) -> torch.Tensor | None:
        """Select the predicted-frame tokens of the layer-k vision hidden state (packing order) and project them.

        Returns ``[N, D_t]`` in the same row order as :meth:`compute_targets`, or ``None`` when empty.
        """
        grids = split_flat_tokens(vision_hidden, token_shapes)  # list of [T,H,W,D]
        nfi_list = self._prepare_noisy_indexes(noisy_frame_indexes, vision_hidden.device)
        preds = [
            grid.index_select(0, nfi).reshape(-1, grid.shape[-1]) for grid, nfi in zip(grids, nfi_list) if nfi.numel() > 0
        ]
        if not preds:
            return None
        out = self.projector(torch.cat(preds, dim=0))  # [N, S*D_t]
        return out.reshape(-1, self.teacher_embed_dim)  # [N*S, D_t] (sub-cell-minor, same row order as the targets)

    def forward(
        self,
        vision_hidden: torch.Tensor,
        token_shapes: Sequence[Sequence[int]],
        noisy_frame_indexes: Sequence[torch.Tensor],
        teacher_tokens: torch.Tensor | Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor | list[int]]:
        """Convenience composition of :meth:`compute_targets` + :meth:`project` (tests / single-call use)."""
        targets = self.compute_targets(teacher_tokens, token_shapes, noisy_frame_indexes)
        if targets is None:
            probe = self.probe(vision_hidden.device, vision_hidden.dtype)
            return {"pred": probe.view(1, 1), "target": probe.detach().view(1, 1) + 1.0, "num_tokens_per_sample": [0] * len(token_shapes), "empty": True}
        target, counts = targets
        pred = self.project(vision_hidden, token_shapes, noisy_frame_indexes)
        assert pred is not None
        return {"pred": pred, "target": target, "num_tokens_per_sample": counts, "empty": False}

    def _per_view_grid(self, t_len: int, h_tok: int, w_tok: int) -> Grid3:
        if w_tok % self.num_views != 0:
            raise ValueError(
                f"MoT token width {w_tok} is not divisible by num_views={self.num_views}; the REPA target assumes "
                "the camera views are concatenated side by side (concat_view)."
            )
        if t_len < 2:
            raise ValueError(f"Need at least 2 latent frames (1 clean + >=1 predicted) for REPA, got {t_len}")
        return (t_len - 1, h_tok, w_tok // self.num_views)
