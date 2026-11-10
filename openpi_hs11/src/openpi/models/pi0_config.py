import dataclasses
import math
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class LanguageRankingConfig:
    """Action-language hinge ranking, ported from cosmos_hs11 (``[model.language_ranking]``,
    ``algorithm/loss/language_ranking.py``; the defaults reproduce the v23/v41 recipe).

    Every training sample is predicted twice from the SAME noised actions and flow time: under its own instruction
    (``E_pos``) and under a uniformly drawn instruction of ANOTHER task of the suite (``E_neg``, see
    ``hs11.libero_v3.NegativePrompt``). ``E`` is the fp32 MSE over the chunk and the real (non-padded) action dims, without
    the main-loss weighting. The per-sample hinge ``relu(margin + E_pos - E_neg)`` (zero for samples whose own prompt was
    CFG-dropped) is averaged over the batch and added as ``loss_weight * hinge`` to openpi's flow-matching loss.

    Weight bookkeeping: Cosmos adds the hinge with ``loss_weight = 1`` next to ``10 x`` an action loss that is a mean over
    the real dims only (ranking : action = 1 : 10 per real action element). openpi's action loss has weight 1 but averages
    over all ``action_dim`` (32) padded slots, so the Cosmos-equivalent weight is ``0.1 * raw_action_dim / action_dim``
    (``0.1 * 7 / 32 = 0.022`` for the 7-D LIBERO actions). ``margin`` is in the same per-element MSE units as in Cosmos.
    """

    enabled: bool = False
    loss_weight: float = 0.1
    margin: float = 0.1
    # Real action dims = the first ``raw_action_dim`` slots. None = infer per batch (slots that are zero in every
    # sample and chunk step of the batch are padding).
    raw_action_dim: int | None = None
    # The gradient through E_neg is multiplied by this (1 = original hinge, 0 = positive-only: the model can satisfy the
    # hinge only by lowering E_pos, never by making the wrong-instruction prediction worse).
    negative_grad_scale: float = 1.0
    # "absolute": E_neg >= E_pos + margin.  "relative": E_neg >= (1 + margin) * E_pos (margin * stop_gradient(E_pos)).
    margin_mode: str = "absolute"
    # 0 = off. Otherwise only this fraction of the VALID pairs -- the easiest ones, i.e. the largest detached
    # ``gap - margin_i`` -- receives the hinge, normalized by ``batch * fraction`` (cosmos_hs11 v24 self-paced ranking).
    self_paced_fraction: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError(f"language_ranking.margin must be finite and >= 0, got {self.margin}")
        if not math.isfinite(self.loss_weight) or self.loss_weight < 0:
            raise ValueError(f"language_ranking.loss_weight must be finite and >= 0, got {self.loss_weight}")
        if not math.isfinite(self.negative_grad_scale) or not 0.0 <= self.negative_grad_scale <= 1.0:
            raise ValueError(f"language_ranking.negative_grad_scale must be in [0, 1], got {self.negative_grad_scale}")
        if self.margin_mode not in ("absolute", "relative"):
            raise ValueError(f"language_ranking.margin_mode must be 'absolute' or 'relative', got {self.margin_mode!r}")
        if self.raw_action_dim is not None and self.raw_action_dim <= 0:
            raise ValueError(f"language_ranking.raw_action_dim must be positive or None, got {self.raw_action_dim}")
        if not math.isfinite(self.self_paced_fraction) or not 0.0 <= self.self_paced_fraction <= 1.0:
            raise ValueError(f"language_ranking.self_paced_fraction must be in [0, 1], got {self.self_paced_fraction}")


@dataclasses.dataclass(frozen=True)
class RepaConfig:
    """DINOv2 representation alignment (REPA, Yu et al. 2024 "Representation Alignment for Generation") on the PaliGemma
    image tokens, ported from the cosmos_hs10 ``[model.repa]`` module (``src/openpi/hs11/repa.py``).

    Student: the residual stream of the image tokens after ``layer_index`` PaliGemma blocks of the SAME forward pass that
    predicts the actions (openpi feeds clean images, so the frames the student sees are the model input, not noised
    frames as in the Cosmos video model). Teacher: frozen DINOv2 patch tokens (post-LN, CLS dropped) of the same
    augmented frames, resized onto the SigLIP token grid (16x16 at 224 px). A projector maps the student tokens to the
    teacher width and two per-sample terms are added to the flow-matching loss, averaged over the valid views' tokens
    (masked image slots such as LIBERO's empty right wrist are excluded):

      ``loss_weight          * (1 - cos(proj(h), dino))``                    token-wise cosine alignment (REPA)
      ``relation_loss_weight * dist(R(proj(h)), R(dino))``                   R(x) = normalize(x) normalize(x)^T over ALL
                                                                             image tokens of the sample (all views), i.e.
                                                                             the (N*T)x(N*T) cosine-similarity maps

    ``mean(chunked_loss) == mean_fm + loss_weight * mean_batch(1 - cos) + relation_loss_weight * mean_batch(rel)``.
    Scale note: openpi's action loss averages over all 32 padded slots (~7/32 of a real-dim mean), so a weight of 0.02
    here corresponds to Cosmos' 1.0 next to its 10x action loss (same bookkeeping as LanguageRankingConfig).
    """

    enabled: bool = False
    # Weight of the (unweighted, in [0, 2]) token cosine loss ``1 - cos``.
    loss_weight: float = 0.0
    # Weight of the relation (similarity-map) loss; "l1" = mean absolute, "l2" = mean squared entry-wise difference.
    relation_loss_weight: float = 0.0
    relation_distance: str = "l1"
    # Number of PaliGemma blocks applied before the aligned hidden state is read (REPA "depth", 1-based; gemma_2b has 18,
    # so 6 ~ REPA's 8 of 28). ``layer_index == depth`` is the residual stream before the final norm.
    layer_index: int = 6
    # Teacher: "dinov2_vitb14" (facebook/dinov2-base, D=768), "dinov2_vits14" (384) or "dinov2_vitl14" (1024).
    teacher: str = "dinov2_vitb14"
    # Explicit ``model.safetensors`` (HF layout) or HF snapshot dir; None = HF cache lookup, then snapshot_download.
    teacher_checkpoint_path: str | None = None
    # Square teacher input (multiple of 14). 224 -> 16x16 patches = the SigLIP grid; other sizes are resized onto it.
    teacher_input_size: int = 224
    # Compute dtype of the frozen teacher ("bfloat16" | "float32"); the loss is always computed in float32.
    teacher_dtype: str = "bfloat16"
    # "mlp" = REPA's Linear-SiLU-Linear-SiLU-Linear projector, "linear" = a single Linear.
    projector_type: str = "mlp"
    projector_hidden_dim: int = 2048
    eps: float = 1e-6

    def __post_init__(self):
        for name in ("loss_weight", "relation_loss_weight", "eps"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"repa.{name} must be finite and >= 0, got {v}")
        if self.relation_distance not in ("l1", "l2"):
            raise ValueError(f"repa.relation_distance must be 'l1' or 'l2', got {self.relation_distance!r}")
        if self.layer_index < 1:
            raise ValueError(f"repa.layer_index is 1-based (blocks applied before the capture), got {self.layer_index}")
        if self.teacher_input_size <= 0 or self.teacher_input_size % 14 != 0:
            raise ValueError(f"repa.teacher_input_size must be a positive multiple of 14, got {self.teacher_input_size}")
        if self.teacher_dtype not in ("bfloat16", "float32"):
            raise ValueError(f"repa.teacher_dtype must be 'bfloat16' or 'float32', got {self.teacher_dtype!r}")
        if self.projector_type not in ("mlp", "linear"):
            raise ValueError(f"repa.projector_type must be 'mlp' or 'linear', got {self.projector_type!r}")
        if self.projector_hidden_dim <= 0:
            raise ValueError(f"repa.projector_hidden_dim must be positive, got {self.projector_hidden_dim}")
        if self.enabled and self.loss_weight == 0.0 and self.relation_loss_weight == 0.0:
            raise ValueError("repa is enabled but both loss_weight and relation_loss_weight are 0")


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    # cosmos_hs11 language ranking (off by default; LIBERO post-training TOMLs enable it via [model.language_ranking]).
    language_ranking: LanguageRankingConfig = dataclasses.field(default_factory=LanguageRankingConfig)
    # cosmos_hs11 DINOv2 REPA distillation (off by default; LIBERO post-training TOMLs enable it via [model.repa]).
    repa: RepaConfig = dataclasses.field(default_factory=RepaConfig)

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
