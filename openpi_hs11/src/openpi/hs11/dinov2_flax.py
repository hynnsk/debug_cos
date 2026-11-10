"""Frozen DINOv2 ViT (facebook/dinov2-*) in plain JAX, the teacher of the cosmos_hs11 REPA distillation.

Why not ``transformers.FlaxDinov2Model``: the teacher has to run INSIDE the jitted, FSDP-sharded openpi train step on the
augmented images of the current batch, with its parameters passed in as a replicated pytree. A ~120-line functional ViT
(patch embedding, CLS + interpolated position embedding, pre-LN blocks with LayerScale, final LayerNorm) that reads the
Hugging Face ``model.safetensors`` directly is simpler to keep correct than threading a linen model through the nnx
bridge. ``dinov2_flax_test.py`` checks it against ``transformers.Dinov2Model`` to 1e-3.

Teacher output (``patch_tokens``) = DINOv2's ``x_norm_patchtokens``: the post-LayerNorm patch tokens, CLS dropped, row
major over the ``input_size / 14`` grid, float32. Input: RGB in ``[0, 1]`` at a multiple of 14 px; ImageNet mean/std are
applied here. The 1370-position (518 px) pretraining position embedding is interpolated ONCE at load time exactly like
``transformers`` 4.53 (``Dinov2Embeddings.interpolate_pos_encoding``: torch bicubic, ``align_corners=False``), which is
the teacher the cosmos_hs10 REPA runs used.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger("openpi")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclasses.dataclass(frozen=True)
class DINOv2Spec:
    name: str
    hf_repo: str
    embed_dim: int
    depth: int
    num_heads: int
    patch_size: int = 14
    mlp_ratio: int = 4
    layer_norm_eps: float = 1e-6


TEACHERS: dict[str, DINOv2Spec] = {
    "dinov2_vits14": DINOv2Spec("dinov2_vits14", "facebook/dinov2-small", 384, 12, 6),
    "dinov2_vitb14": DINOv2Spec("dinov2_vitb14", "facebook/dinov2-base", 768, 12, 12),
    "dinov2_vitl14": DINOv2Spec("dinov2_vitl14", "facebook/dinov2-large", 1024, 24, 16),
}
ALIASES: dict[str, str] = {
    "dinov2_small": "dinov2_vits14",
    "dinov2-small": "dinov2_vits14",
    "dinov2_base": "dinov2_vitb14",
    "dinov2-base": "dinov2_vitb14",
    "dinov2_large": "dinov2_vitl14",
    "dinov2-large": "dinov2_vitl14",
}


def resolve_spec(name: str) -> DINOv2Spec:
    key = ALIASES.get(name.lower(), name.lower())
    if key not in TEACHERS:
        raise ValueError(f"Unknown DINOv2 teacher {name!r}; expected one of {sorted(TEACHERS)} (or an alias {sorted(ALIASES)})")
    return TEACHERS[key]


# ---------------------------------------------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------------------------------------------
def _hf_cache_roots() -> list[pathlib.Path]:
    roots = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(pathlib.Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(pathlib.Path(os.environ["HF_HOME"]) / "hub")
    storage = os.environ.get("COSMOS_STORAGE", str(pathlib.Path.home() / "project" / "cosmos_storage"))
    roots.append(pathlib.Path(storage) / "hf_cache" / "hub")
    roots.append(pathlib.Path.home() / ".cache" / "huggingface" / "hub")
    return [r.expanduser() for r in roots]


def resolve_checkpoint(spec: DINOv2Spec, path: str | None = None) -> pathlib.Path:
    """``model.safetensors`` of the teacher: an explicit file / snapshot dir, else the HF cache (``$HF_HUB_CACHE``,
    ``$HF_HOME/hub``, ``$COSMOS_STORAGE/hf_cache/hub``, ``~/.cache/huggingface/hub``), else ``snapshot_download``."""
    # Snapshot paths are returned as-is (model.safetensors is a symlink into blobs/; the snapshot dir also holds config.json).
    if path:
        p = pathlib.Path(path).expanduser()
        if p.is_dir():
            p = p / "model.safetensors"
        if not p.is_file():
            raise FileNotFoundError(f"DINOv2 teacher checkpoint not found: {path} (expected a model.safetensors file or its directory)")
        return p
    if not spec.hf_repo:
        raise ValueError(f"teacher {spec.name!r} has no Hugging Face repo; pass teacher_checkpoint_path")
    repo_dir = f"models--{spec.hf_repo.replace('/', '--')}"
    for root in _hf_cache_roots():
        hits = sorted((root / repo_dir / "snapshots").glob("*/model.safetensors"))
        if hits:
            return hits[-1]
    from huggingface_hub import snapshot_download

    logger.info(f"DINOv2 teacher {spec.hf_repo} not in the HF cache -> snapshot_download")
    snapshot = snapshot_download(spec.hf_repo, allow_patterns=["model.safetensors", "config.json"])
    return pathlib.Path(snapshot) / "model.safetensors"


def load_hf_state(path: str | os.PathLike) -> dict[str, np.ndarray]:
    from safetensors import safe_open

    with safe_open(str(path), framework="np") as f:
        return {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118


def interpolate_pos_embed(pos: np.ndarray, grid: int) -> np.ndarray:
    """``[1, 1 + M*M, D]`` -> ``[1, 1 + grid*grid, D]`` like ``transformers.Dinov2Embeddings.interpolate_pos_encoding``
    (torch bicubic on the MxM patch grid, ``align_corners=False``, no antialias; the CLS position is kept)."""
    n = pos.shape[1] - 1
    m = round(n**0.5)
    if m * m != n:
        raise ValueError(f"position embedding has {n} patch positions, not a square grid")
    if m == grid:
        return np.asarray(pos, dtype=np.float32)
    import torch
    import torch.nn.functional as F  # noqa: N812

    cls, patch = pos[:, :1], pos[:, 1:]
    t = torch.from_numpy(np.ascontiguousarray(patch, dtype=np.float32)).reshape(1, m, m, -1).permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(grid, grid), mode="bicubic", align_corners=False)
    patch = t.permute(0, 2, 3, 1).reshape(1, grid * grid, -1).numpy()
    return np.concatenate([np.asarray(cls, dtype=np.float32), patch], axis=1)


def convert_hf_state(sd: dict[str, np.ndarray], spec: DINOv2Spec, input_size: int, dtype=jnp.bfloat16) -> dict:
    """Hugging Face ``Dinov2Model`` state dict -> the pytree ``patch_tokens`` expects (blocks stacked along axis 0 for
    ``lax.scan``; Linear weights transposed to ``[in, out]``; q/k/v fused). Matmul weights are cast to ``dtype``,
    everything else stays float32."""
    if any(k.startswith("embeddings.register_tokens") for k in sd):
        raise ValueError("DINOv2 checkpoints with register tokens are not supported (use facebook/dinov2-{small,base,large})")
    d, p, grid = spec.embed_dim, spec.patch_size, input_size // spec.patch_size
    f32 = np.float32

    def lin(name: str) -> np.ndarray:  # torch [out, in] -> [in, out]
        return np.ascontiguousarray(sd[name].T)

    def stacked(fn):
        return np.stack([fn(i) for i in range(spec.depth)])

    def cast(x):
        return jnp.asarray(x, dtype=dtype)

    conv = sd["embeddings.patch_embeddings.projection.weight"]  # [D, 3, p, p]
    if conv.shape != (d, 3, p, p):
        raise ValueError(f"patch embedding weight {conv.shape} does not match {spec}")
    return {
        # image patch flattened in (h, w, c) order -> kernel [p, p, 3, D] -> [p*p*3, D]
        "patch_kernel": cast(conv.transpose(2, 3, 1, 0).reshape(p * p * 3, d)),
        "patch_bias": jnp.asarray(sd["embeddings.patch_embeddings.projection.bias"], f32),
        "cls_token": jnp.asarray(sd["embeddings.cls_token"].reshape(1, 1, d), f32),
        "pos_embed": jnp.asarray(interpolate_pos_embed(sd["embeddings.position_embeddings"], grid), f32),
        "blocks": {
            "norm1_scale": jnp.asarray(stacked(lambda i: sd[f"encoder.layer.{i}.norm1.weight"]), f32),
            "norm1_bias": jnp.asarray(stacked(lambda i: sd[f"encoder.layer.{i}.norm1.bias"]), f32),
            "qkv_kernel": cast(
                stacked(
                    lambda i: np.concatenate(
                        [lin(f"encoder.layer.{i}.attention.attention.{n}.weight") for n in ("query", "key", "value")], axis=1
                    )
                )
            ),
            "qkv_bias": jnp.asarray(
                stacked(
                    lambda i: np.concatenate(
                        [sd[f"encoder.layer.{i}.attention.attention.{n}.bias"] for n in ("query", "key", "value")], axis=0
                    )
                ),
                f32,
            ),
            "proj_kernel": cast(stacked(lambda i: lin(f"encoder.layer.{i}.attention.output.dense.weight"))),
            "proj_bias": jnp.asarray(stacked(lambda i: sd[f"encoder.layer.{i}.attention.output.dense.bias"]), f32),
            "ls1": jnp.asarray(stacked(lambda i: sd[f"encoder.layer.{i}.layer_scale1.lambda1"]), f32),
            "norm2_scale": jnp.asarray(stacked(lambda i: sd[f"encoder.layer.{i}.norm2.weight"]), f32),
            "norm2_bias": jnp.asarray(stacked(lambda i: sd[f"encoder.layer.{i}.norm2.bias"]), f32),
            "fc1_kernel": cast(stacked(lambda i: lin(f"encoder.layer.{i}.mlp.fc1.weight"))),
            "fc1_bias": jnp.asarray(stacked(lambda i: sd[f"encoder.layer.{i}.mlp.fc1.bias"]), f32),
            "fc2_kernel": cast(stacked(lambda i: lin(f"encoder.layer.{i}.mlp.fc2.weight"))),
            "fc2_bias": jnp.asarray(stacked(lambda i: sd[f"encoder.layer.{i}.mlp.fc2.bias"]), f32),
            "ls2": jnp.asarray(stacked(lambda i: sd[f"encoder.layer.{i}.layer_scale2.lambda1"]), f32),
        },
        "norm_scale": jnp.asarray(sd["layernorm.weight"], f32),
        "norm_bias": jnp.asarray(sd["layernorm.bias"], f32),
    }


def random_params(spec: DINOv2Spec, key, input_size: int, dtype=jnp.float32) -> dict:
    """Randomly initialized teacher of the given architecture (tests / smokes without the real weights)."""
    d, p, grid, h = spec.embed_dim, spec.patch_size, input_size // spec.patch_size, spec.embed_dim * spec.mlp_ratio
    keys = iter(jax.random.split(key, 16))

    def normal(shape, scale=0.02):
        return scale * jax.random.normal(next(keys), shape, jnp.float32)

    n = grid * grid
    blocks = {
        "norm1_scale": jnp.ones((spec.depth, d)),
        "norm1_bias": jnp.zeros((spec.depth, d)),
        "qkv_kernel": normal((spec.depth, d, 3 * d)).astype(dtype),
        "qkv_bias": jnp.zeros((spec.depth, 3 * d)),
        "proj_kernel": normal((spec.depth, d, d)).astype(dtype),
        "proj_bias": jnp.zeros((spec.depth, d)),
        "ls1": jnp.full((spec.depth, d), 0.5),
        "norm2_scale": jnp.ones((spec.depth, d)),
        "norm2_bias": jnp.zeros((spec.depth, d)),
        "fc1_kernel": normal((spec.depth, d, h)).astype(dtype),
        "fc1_bias": jnp.zeros((spec.depth, h)),
        "fc2_kernel": normal((spec.depth, h, d)).astype(dtype),
        "fc2_bias": jnp.zeros((spec.depth, d)),
        "ls2": jnp.full((spec.depth, d), 0.5),
    }
    return {
        "patch_kernel": normal((p * p * 3, d)).astype(dtype),
        "patch_bias": jnp.zeros((d,)),
        "cls_token": normal((1, 1, d)),
        "pos_embed": normal((1, 1 + n, d)),
        "blocks": blocks,
        "norm_scale": jnp.ones((d,)),
        "norm_bias": jnp.zeros((d,)),
    }


# ---------------------------------------------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------------------------------------------
def _layer_norm(x, scale, bias, eps):
    x = x.astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * scale + bias


def patch_tokens(params: dict, spec: DINOv2Spec, images) -> jax.Array:
    """``images``: float RGB in ``[0, 1]``, ``[b, S, S, 3]`` with ``S`` a multiple of ``spec.patch_size`` matching the
    grid the position embedding was interpolated for. Returns the post-LN patch tokens ``[b, (S/14)^2, D]`` (float32),
    row major over the patch grid."""
    b, h, w, c = images.shape
    p, d, nh = spec.patch_size, spec.embed_dim, spec.num_heads
    if h % p or w % p:
        raise ValueError(f"teacher input {h}x{w} must be a multiple of the patch size {p}")
    gh, gw = h // p, w // p
    if params["pos_embed"].shape[1] != 1 + gh * gw:
        raise ValueError(
            f"position embedding holds {params['pos_embed'].shape[1] - 1} patch positions but the input gives {gh * gw}; "
            "load the teacher with the input size it is used at"
        )
    cdt = params["patch_kernel"].dtype
    f32 = jnp.float32
    hd = d // nh

    x = (jnp.asarray(images, f32) - jnp.asarray(IMAGENET_MEAN, f32)) / jnp.asarray(IMAGENET_STD, f32)
    x = x.reshape(b, gh, p, gw, p, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, gh * gw, p * p * c)
    x = jnp.dot(x.astype(cdt), params["patch_kernel"], preferred_element_type=f32) + params["patch_bias"]
    cls = jnp.broadcast_to(params["cls_token"], (b, 1, d))
    x = jnp.concatenate([cls, x], axis=1) + params["pos_embed"]
    n = x.shape[1]

    def block(x, q):
        hs = _layer_norm(x, q["norm1_scale"], q["norm1_bias"], spec.layer_norm_eps)
        qkv = jnp.dot(hs.astype(cdt), q["qkv_kernel"], preferred_element_type=f32) + q["qkv_bias"]
        qq, kk, vv = (t.reshape(b, n, nh, hd) for t in jnp.split(qkv, 3, axis=-1))
        logits = jnp.einsum("bqhd,bkhd->bhqk", qq.astype(cdt), kk.astype(cdt), preferred_element_type=f32) * (hd**-0.5)
        probs = jax.nn.softmax(logits, axis=-1)
        attn = jnp.einsum("bhqk,bkhd->bqhd", probs.astype(cdt), vv.astype(cdt), preferred_element_type=f32).reshape(b, n, d)
        attn = jnp.dot(attn.astype(cdt), q["proj_kernel"], preferred_element_type=f32) + q["proj_bias"]
        x = x + q["ls1"] * attn
        hs = _layer_norm(x, q["norm2_scale"], q["norm2_bias"], spec.layer_norm_eps)
        hs = jnp.dot(hs.astype(cdt), q["fc1_kernel"], preferred_element_type=f32) + q["fc1_bias"]
        hs = jax.nn.gelu(hs, approximate=False)
        hs = jnp.dot(hs.astype(cdt), q["fc2_kernel"], preferred_element_type=f32) + q["fc2_bias"]
        x = x + q["ls2"] * hs
        return x, None

    x, _ = jax.lax.scan(block, x, params["blocks"])
    x = _layer_norm(x, params["norm_scale"], params["norm_bias"], spec.layer_norm_eps)
    return x[:, 1:]
