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
