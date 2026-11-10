"""cosmos_hs11 DINOv2 REPA distillation for openpi (``pi0_config.RepaConfig``, ``[model.repa]`` in the LIBERO TOMLs).

Pieces (mirroring the cosmos_hs10 ``model/generator/repa`` module where the recipe comes from):

* ``RepaProjector`` -- the trainable student head (REPA's ``Linear-SiLU-Linear-SiLU-Linear`` MLP or one Linear), an nnx
  submodule of ``Pi0`` (``repa_proj``), so it is trained, EMA'd and checkpointed with the policy. Checkpoints without it
  (pi05_base, Reptile thetas) load fine: ``weight_loaders.CheckpointWeightLoader`` keeps the fresh init for ``repa_proj``.
* ``repa_losses`` -- the two per-sample terms: ``1 - cos`` between projected student and teacher tokens, and the entry-wise
  "l1" / "l2" distance between the pairwise cosine-similarity maps ``R(x) = normalize(x) normalize(x)^T`` of the two token
  sets (all image tokens of a sample, every camera view, so within- and cross-view pairs alike). A token mask drops the
  tokens of masked image slots from both terms (LIBERO's empty right wrist).
* ``make_teacher_fn`` -- the frozen DINOv2 (``dinov2_flax``) applied to the SAME augmented ``[-1, 1]`` images SigLIP sees,
  output resized onto the SigLIP 16x16 token grid when the teacher grid differs (``teacher_input_size != 224``).
* ``load_teacher_for_training`` -- used by ``scripts/train.py``: the teacher parameters are NOT part of the model / train
  state (never trained, never checkpointed); they are an extra replicated argument of the jitted train step.
"""

from __future__ import annotations

from collections.abc import Callable
import logging

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.hs11 import dinov2_flax as _dino
from openpi.models import model as _model

logger = logging.getLogger("openpi")

SIGLIP_PATCH = 14  # openpi's image encoder is SigLIP So400m/14 at IMAGE_RESOLUTION (224 -> 16x16 tokens per view)


def student_grid() -> tuple[int, int]:
    """The SigLIP token grid of one camera view: every teacher target is resized onto it."""
    return (_model.IMAGE_RESOLUTION[0] // SIGLIP_PATCH, _model.IMAGE_RESOLUTION[1] // SIGLIP_PATCH)


def teacher_spec(cfg) -> _dino.DINOv2Spec:
    return _dino.resolve_spec(cfg.teacher)


class RepaProjector(nnx.Module):
    """Student head: ``mlp`` = REPA's ``build_mlp`` (Linear-SiLU-Linear-SiLU-Linear), ``linear`` = one Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, projector_type: str = "mlp", *, rngs: nnx.Rngs):
        if projector_type not in ("mlp", "linear"):
            raise ValueError(f"projector_type must be 'mlp' or 'linear', got {projector_type!r}")
        self.projector_type = projector_type
        if projector_type == "linear":
            self.fc1 = nnx.Linear(in_dim, out_dim, rngs=rngs)
        else:
            self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
            self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
            self.fc3 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, x):
        x = x.astype(jnp.float32)
        if self.projector_type == "linear":
            return self.fc1(x)
        x = nnx.silu(self.fc1(x))
        x = nnx.silu(self.fc2(x))
        return self.fc3(x)


def l2_normalize(x, eps: float = 1e-6):
    """Row-wise ``x / max(||x||, eps)`` (``torch.nn.functional.normalize`` semantics), float32."""
    x = x.astype(jnp.float32)
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def similarity_map(x, eps: float = 1e-6):
    """Pairwise cosine similarities of the rows of ``x`` (``[b, n, d] -> [b, n, n]``), full float32 precision."""
    xn = l2_normalize(x, eps)
    return jnp.einsum("bnd,bmd->bnm", xn, xn, precision=jax.lax.Precision.HIGHEST)


def repa_losses(pred, target, token_mask, *, relation_distance: str = "l1", eps: float = 1e-6):
    """Per-sample REPA terms over the valid tokens.

    ``pred`` / ``target``: ``[b, n, d]`` (projected student / teacher tokens, same layout); ``token_mask``: ``[b, n]``
    (1 = token counts). Returns ``(cos_loss, cos_mean, rel_loss)``, each ``[b]``:
    ``cos_loss = 1 - mean_valid cos(pred_i, target_i)``, ``rel_loss = mean over valid pairs (i, j) of
    |R(pred)_ij - R(target)_ij|`` ("l1") or its square ("l2"), ``R`` the cosine-similarity map of the sample's tokens.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {pred.shape} and target {target.shape} must have the same shape")
    if relation_distance not in ("l1", "l2"):
        raise ValueError(f"relation_distance must be 'l1' or 'l2', got {relation_distance!r}")
    token_mask = token_mask.astype(jnp.float32)
    pn, tn = l2_normalize(pred, eps), l2_normalize(target, eps)
    cos = jnp.sum(pn * tn, axis=-1)  # [b, n]
    n_valid = jnp.maximum(jnp.sum(token_mask, axis=-1), 1.0)
    cos_mean = jnp.sum(cos * token_mask, axis=-1) / n_valid
    cos_loss = 1.0 - cos_mean

    diff = similarity_map(pred, eps) - similarity_map(target, eps)  # [b, n, n]
    penalty = jnp.abs(diff) if relation_distance == "l1" else jnp.square(diff)
    pair_mask = token_mask[:, :, None] * token_mask[:, None, :]
    rel_loss = jnp.sum(penalty * pair_mask, axis=(-2, -1)) / jnp.maximum(jnp.sum(pair_mask, axis=(-2, -1)), 1.0)
    return cos_loss, cos_mean, rel_loss


def resize_token_grid(tokens, src_grid: tuple[int, int], dst_grid: tuple[int, int]):
    """``[b, gh*gw, d]`` row-major tokens on ``src_grid`` -> bilinear (antialiased when shrinking) resample onto
    ``dst_grid``; identity when the grids match."""
    if tuple(src_grid) == tuple(dst_grid):
        return tokens
    b, n, d = tokens.shape
    if n != src_grid[0] * src_grid[1]:
        raise ValueError(f"{n} tokens do not form a {src_grid} grid")
    grid = tokens.reshape(b, src_grid[0], src_grid[1], d)
    grid = jax.image.resize(grid, (b, dst_grid[0], dst_grid[1], d), method="bilinear", antialias=True)
    return grid.reshape(b, dst_grid[0] * dst_grid[1], d)


def make_teacher_fn(cfg, params, grid: tuple[int, int] | None = None) -> Callable:
    """``fn(images)``: openpi ``[-1, 1]`` float images ``[b, h, w, 3]`` (the augmented model input) -> DINOv2 patch
    tokens ``[b, gh*gw, D]`` float32 on the student ``grid`` (default: the SigLIP grid), stop-gradient'ed."""
    spec = teacher_spec(cfg)
    size = int(cfg.teacher_input_size)
    teacher_grid = (size // spec.patch_size, size // spec.patch_size)
    grid = student_grid() if grid is None else tuple(grid)

    def fn(images):
        x = (jnp.asarray(images, jnp.float32) + 1.0) / 2.0
        x = jnp.clip(x, 0.0, 1.0)
        if x.shape[1:3] != (size, size):
            x = jax.image.resize(x, (x.shape[0], size, size, x.shape[3]), method="bilinear", antialias=True)
        tokens = _dino.patch_tokens(params, spec, x)
        tokens = resize_token_grid(tokens, teacher_grid, grid)
        return jax.lax.stop_gradient(tokens)

    return fn


def load_teacher_params(cfg) -> dict:
    """The frozen DINOv2 parameters as a host pytree (matmul weights in ``cfg.teacher_dtype``)."""
    spec = teacher_spec(cfg)
    path = _dino.resolve_checkpoint(spec, cfg.teacher_checkpoint_path)
    dtype = jnp.bfloat16 if cfg.teacher_dtype == "bfloat16" else jnp.float32
    params = _dino.convert_hf_state(_dino.load_hf_state(path), spec, cfg.teacher_input_size, dtype=dtype)
    n_params = sum(int(x.size) for x in jax.tree.leaves(params))
    logger.info(f"REPA teacher {spec.name} ({n_params / 1e6:.1f} M params, {cfg.teacher_dtype}) loaded from {path}")
    return params


def load_teacher_for_training(config, sharding) -> dict | None:
    """``scripts/train.py``: the replicated teacher pytree when ``config.model.repa.enabled``, else None."""
    repa = getattr(config.model, "repa", None)
    if repa is None or not repa.enabled:
        return None
    return jax.device_put(load_teacher_params(repa), sharding)
