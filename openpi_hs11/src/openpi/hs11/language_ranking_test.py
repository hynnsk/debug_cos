"""cosmos_hs11 language ranking ported to openpi: config, data transforms, TOML knob and the paired hinge in pi0 (CPU, dummy pi05)."""

import dataclasses
import os
import pathlib
import random

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.hs11 import libero_v3
from openpi.hs11 import toml_config
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
import openpi.models.pi0_config as pi0_config
import openpi.transforms as _transforms

ROOT = pathlib.Path(__file__).resolve().parents[3]
TOML_DIR = ROOT / "scripts" / "hs11" / "toml"
RAW_DIM = 7
HORIZON = 4
BATCH = 3


def _randomize_zero_kernels(model) -> None:
    # pi05's adaRMSNorm modulation is zero-initialized (adaLN-zero): a fresh model ignores its prefix (prompt), which
    # would make E_pos == E_neg. Randomize every zero kernel deterministically so the prompt matters (cf. cfg_test).
    state = nnx.state(model)
    flat = state.flat_state()
    key = jax.random.key(42)
    for _, var in flat.items():
        arr = var.value
        if hasattr(arr, "ndim") and arr.ndim >= 2 and float(jnp.abs(arr).max()) == 0.0:
            key, sub = jax.random.split(key)
            var.value = 0.05 * jax.random.normal(sub, arr.shape, arr.dtype)
    nnx.update(model, nnx.State.from_flat_path(flat))


def _config(**ranking) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=HORIZON,
        discrete_state_input=False,
        language_ranking=pi0_config.LanguageRankingConfig(**ranking),
    )


def _model_for(config: pi0_config.Pi0Config):
    model = config.create(jax.random.key(0))  # same key -> identical parameters for every ranking config
    _randomize_zero_kernels(model)
    return model


@pytest.fixture(scope="module")
def batch():
    config = _config()
    obs, _ = config.inputs_spec(batch_size=BATCH)
    rng = jax.random.key(1)
    obs = jax.tree.map(
        lambda spec: jax.random.uniform(rng, spec.shape, minval=-1, maxval=1) if spec.dtype == jnp.float32 else jnp.zeros(spec.shape, spec.dtype),
        obs,
    )
    tok = _tokenizer.PaligemmaTokenizer(config.max_token_len)
    pos = [tok.tokenize("put the bowl on the plate") for _ in range(BATCH)]
    neg = [tok.tokenize(p) for p in ("open the top drawer", "turn on the stove", "put the book in the caddy")]
    obs = obs.replace(
        tokenized_prompt=jnp.asarray(np.stack([t for t, _ in pos])),
        tokenized_prompt_mask=jnp.asarray(np.stack([m for _, m in pos])),
        tokenized_negative_prompt=jnp.asarray(np.stack([t for t, _ in neg])),
        tokenized_negative_prompt_mask=jnp.asarray(np.stack([m for _, m in neg])),
        ranking_valid=jnp.asarray([True, True, False]),
    )
    actions = jax.random.normal(jax.random.key(2), (BATCH, HORIZON, config.action_dim))
    actions = actions.at[..., RAW_DIM:].set(0.0)  # PadStatesAndActions zero-pads the 7 real dims to 32
    return obs, actions


def _closed_form(model, obs, actions, rng, cfg: pi0_config.LanguageRankingConfig):
    """Recompute the plain chunk loss, E_pos / E_neg and the hinge from the model's own velocity predictions."""
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, obs, train=False)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
    u_t = noise - actions
    v_pos, _ = model._predict_velocity(observation, x_t, time)  # noqa: SLF001
    v_neg, _ = model._predict_velocity(  # noqa: SLF001
        observation.replace(tokenized_prompt=obs.tokenized_negative_prompt, tokenized_prompt_mask=obs.tokenized_negative_prompt_mask),
        x_t,
        time,
    )
    plain = jnp.mean(jnp.square(v_pos - u_t), axis=-1)
    e_pos = jnp.mean(jnp.square(v_pos - u_t)[..., :RAW_DIM], axis=(-2, -1))
    e_neg = jnp.mean(jnp.square(v_neg - u_t)[..., :RAW_DIM], axis=(-2, -1))
    margins = cfg.margin * e_pos if cfg.margin_mode == "relative" else cfg.margin
    hinge = jax.nn.relu(margins + e_pos - e_neg)
    return plain, e_pos, e_neg, hinge


def test_disabled_ranking_is_the_plain_flow_matching_loss(batch):
    obs, actions = batch
    model = _model_for(_config(enabled=False))
    rng = jax.random.key(7)
    chunked, metrics = model.compute_loss_and_metrics(rng, obs, actions)
    assert metrics == {}
    plain, *_ = _closed_form(model, obs, actions, rng, _config().language_ranking)
    np.testing.assert_allclose(np.asarray(chunked), np.asarray(plain), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(model.compute_loss(rng, obs, actions)), np.asarray(chunked), rtol=1e-6)


@pytest.mark.parametrize("raw_action_dim", [RAW_DIM, None], ids=["explicit_dim", "inferred_dim"])
def test_hinge_matches_closed_form_and_invalid_pairs_contribute_zero(batch, raw_action_dim):
    obs, actions = batch
    cfg = pi0_config.LanguageRankingConfig(enabled=True, loss_weight=0.5, margin=0.1, raw_action_dim=raw_action_dim)
    model = _model_for(_config(**{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()}))
    rng = jax.random.key(7)
    chunked, metrics = model.compute_loss_and_metrics(rng, obs, actions)
    plain, e_pos, e_neg, hinge = _closed_form(model, obs, actions, rng, cfg)
    valid = np.array([1.0, 1.0, 0.0])
    assert float(jnp.abs(e_pos - e_neg).max()) > 1e-6  # the wrong instruction changes the prediction
    expected = np.asarray(plain) + cfg.loss_weight * (np.asarray(hinge) * valid)[:, None]
    np.testing.assert_allclose(np.asarray(chunked), expected, rtol=1e-5, atol=1e-6)
    # mean(chunked) == mean_fm + loss_weight * mean_batch(hinge * valid): the Cosmos ``total += w * ranking_loss`` contract
    np.testing.assert_allclose(float(jnp.mean(chunked)), float(jnp.mean(plain)) + cfg.loss_weight * float(np.mean(np.asarray(hinge) * valid)), rtol=1e-5)
    np.testing.assert_allclose(float(metrics["language_ranking_loss"]), float(np.mean(np.asarray(hinge) * valid)), rtol=1e-5)
    np.testing.assert_allclose(float(metrics["language_ranking_weighted_loss"]), cfg.loss_weight * float(np.mean(np.asarray(hinge) * valid)), rtol=1e-5)
    np.testing.assert_allclose(float(metrics["language_ranking_positive_error"]), float(np.mean(np.asarray(e_pos)[:2])), rtol=1e-5)
    np.testing.assert_allclose(float(metrics["language_ranking_negative_error"]), float(np.mean(np.asarray(e_neg)[:2])), rtol=1e-5)
    np.testing.assert_allclose(float(metrics["language_ranking_gap"]), float(np.mean(np.asarray(e_neg - e_pos)[:2])), rtol=1e-5)
    assert float(metrics["language_ranking_valid_fraction"]) == pytest.approx(2 / 3)
    satisfied = np.mean((np.asarray(e_neg - e_pos) >= cfg.margin)[:2])
    assert float(metrics["language_ranking_margin_satisfied"]) == pytest.approx(satisfied)
    # the invalid (CFG-dropped) sample is untouched
    np.testing.assert_allclose(np.asarray(chunked)[2], np.asarray(plain)[2], rtol=1e-6)


def test_loss_value_is_independent_of_negative_grad_scale_and_relative_margin_is_scaled(batch):
    obs, actions = batch
    rng = jax.random.key(7)
    full = _model_for(_config(enabled=True, loss_weight=1.0, margin=0.1, raw_action_dim=RAW_DIM, negative_grad_scale=1.0))
    pos_only = _model_for(_config(enabled=True, loss_weight=1.0, margin=0.1, raw_action_dim=RAW_DIM, negative_grad_scale=0.0))
    a, _ = full.compute_loss_and_metrics(rng, obs, actions)
    b, _ = pos_only.compute_loss_and_metrics(rng, obs, actions)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)  # same value, only the gradient path differs
    rel_cfg = pi0_config.LanguageRankingConfig(enabled=True, loss_weight=1.0, margin=0.5, raw_action_dim=RAW_DIM, margin_mode="relative")
    rel = _model_for(_config(**{f.name: getattr(rel_cfg, f.name) for f in rel_cfg.__dataclass_fields__.values()}))
    chunked, _ = rel.compute_loss_and_metrics(rng, obs, actions)
    plain, _, _, hinge = _closed_form(rel, obs, actions, rng, rel_cfg)
    expected = np.asarray(plain) + (np.asarray(hinge) * np.array([1.0, 1.0, 0.0]))[:, None]
    np.testing.assert_allclose(np.asarray(chunked), expected, rtol=1e-5, atol=1e-6)


def test_enabled_without_negative_prompts_raises(batch):
    obs, actions = batch
    model = _model_for(_config(enabled=True))
    bare = obs.replace(tokenized_negative_prompt=None, tokenized_negative_prompt_mask=None, ranking_valid=None)
    with pytest.raises(ValueError, match="tokenized_negative_prompt"):
        model.compute_loss(jax.random.key(0), bare, actions)
    with pytest.raises(ValueError, match="provided together"):
        _model.Observation.from_dict({"image": {}, "image_mask": {}, "state": np.zeros(32), "tokenized_negative_prompt": np.zeros(4, np.int32)})


def test_config_validation():
    with pytest.raises(ValueError, match="margin"):
        pi0_config.LanguageRankingConfig(margin=-0.1)
    with pytest.raises(ValueError, match="negative_grad_scale"):
        pi0_config.LanguageRankingConfig(negative_grad_scale=1.5)
    with pytest.raises(ValueError, match="margin_mode"):
        pi0_config.LanguageRankingConfig(margin_mode="hinge")
    with pytest.raises(ValueError, match="raw_action_dim"):
        pi0_config.LanguageRankingConfig(raw_action_dim=0)
    with pytest.raises(ValueError, match="self_paced_fraction"):
        pi0_config.LanguageRankingConfig(self_paced_fraction=1.5)


def test_self_paced_fraction_ranks_only_the_easiest_valid_pairs(batch):
    obs, actions = batch
    rng = jax.random.key(7)
    base = pi0_config.LanguageRankingConfig(enabled=True, loss_weight=1.0, margin=0.1, raw_action_dim=RAW_DIM)
    as_kw = lambda c: {f.name: getattr(c, f.name) for f in c.__dataclass_fields__.values()}
    full, _ = _model_for(_config(**as_kw(base))).compute_loss_and_metrics(rng, obs, actions)
    # fraction 1.0 selects every valid pair and normalizes by batch * 1 == the plain hinge
    one, m1 = _model_for(_config(**as_kw(dataclasses.replace(base, self_paced_fraction=1.0)))).compute_loss_and_metrics(rng, obs, actions)
    np.testing.assert_allclose(np.asarray(one), np.asarray(full), rtol=1e-6)
    assert float(m1["language_ranking_selected_fraction"]) == pytest.approx(1.0)
    # fraction 0.5 with 2 valid pairs -> n_select = min(2, max(1, ceil(1))) = 1: only the easiest (largest gap - margin)
    half_cfg = dataclasses.replace(base, self_paced_fraction=0.5)
    model = _model_for(_config(**as_kw(half_cfg)))
    half, m_half = model.compute_loss_and_metrics(rng, obs, actions)
    plain, e_pos, e_neg, hinge = _closed_form(model, obs, actions, rng, half_cfg)
    score = np.asarray(e_neg - e_pos) - half_cfg.margin
    score[2] = -np.inf  # invalid pair
    easiest = int(np.argmax(score))
    expected_term = np.zeros(BATCH)
    expected_term[easiest] = float(np.asarray(hinge)[easiest]) / 0.5  # sum / (batch * fraction) once averaged over the batch
    np.testing.assert_allclose(np.asarray(half), np.asarray(plain) + expected_term[:, None], rtol=1e-5, atol=1e-6)
    assert float(m_half["language_ranking_selected_fraction"]) == pytest.approx(0.5)
    np.testing.assert_allclose(float(m_half["language_ranking_loss"]), expected_term.mean(), rtol=1e-5)


def test_negative_prompt_and_tokenizer_transforms():
    instructions = ("a", "b", "c")
    neg = libero_v3.NegativePrompt(instructions=instructions)
    random.seed(0)
    for _ in range(20):
        out = neg({"prompt": "a", "k": 1})
        assert out["negative_prompt"] in {"b", "c"} and out["ranking_valid"] is True and out["k"] == 1
    dropped = neg({"prompt": ""})  # PromptDropout replaced the prompt -> the pair is invalid but still well-formed
    assert dropped["ranking_valid"] is False and dropped["negative_prompt"] in instructions
    assert neg({"k": 1}) == {"k": 1}
    with pytest.raises(ValueError, match="two distinct"):
        libero_v3.NegativePrompt(instructions=("a", "a"))
    # LiberoV3Inputs passes the two keys through
    img = np.zeros((8, 8, 3), np.uint8)
    inputs = libero_v3.LiberoV3Inputs(model_type=_model.ModelType.PI05)(
        {"observation/image": img, "observation/wrist_image": img, "observation/state": np.zeros(8), "prompt": "a", "negative_prompt": "b", "ranking_valid": True}
    )
    assert inputs["negative_prompt"] == "b" and inputs["ranking_valid"] is True
    # TokenizeNegativePrompt mirrors TokenizePrompt and is a no-op without the key
    tok = _tokenizer.PaligemmaTokenizer(16)
    t = _transforms.TokenizeNegativePrompt(tok)
    assert t({"prompt": "x"}) == {"prompt": "x"}
    out = t({"negative_prompt": "b", "ranking_valid": True, "state": np.zeros(8)})
    assert "negative_prompt" not in out and out["tokenized_negative_prompt"].shape == (16,) and out["tokenized_negative_prompt_mask"].shape == (16,)
    assert out["ranking_valid"].dtype == np.bool_ and bool(out["ranking_valid"]) is True
    ref, ref_mask = tok.tokenize("b")
    np.testing.assert_array_equal(out["tokenized_negative_prompt"], ref)
    np.testing.assert_array_equal(out["tokenized_negative_prompt_mask"], ref_mask)


def test_toml_knob_wires_the_negative_stream(tmp_path):
    (tmp_path / "params").mkdir()
    overrides = [f"checkpoint.init_params={tmp_path / 'params'}"]
    # v2.4 = the Cosmos v97-style relative hinge; the Cosmos block (incl. self_paced_fraction) is accepted verbatim
    cfg = toml_config.load_train_config(TOML_DIR / "libero10_3ep_reptile_v2.4.toml", overrides)
    lr = cfg.model.language_ranking
    assert lr.enabled and lr.loss_weight == 0.02 and lr.margin == 0.2 and lr.raw_action_dim == RAW_DIM
    assert lr.margin_mode == "relative" and lr.negative_grad_scale == 1.0 and lr.self_paced_fraction == 0.0
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert [type(t).__name__ for t in dc.repack_transforms.inputs] == ["RepackTransform", "NegativePrompt"]
    assert len(dc.repack_transforms.inputs[1].instructions) == 10
    assert "TokenizeNegativePrompt" in [type(t).__name__ for t in dc.model_transforms.inputs]
    # off by default / CLI override / unknown key / not a Reptile knob
    base = toml_config.load_train_config(TOML_DIR / "goal_3ep_base_v1.toml")
    assert not base.model.language_ranking.enabled
    assert "NegativePrompt" not in [type(t).__name__ for t in base.data.create(base.assets_dirs, base.model).repack_transforms.inputs]
    off = toml_config.load_train_config(TOML_DIR / "libero10_3ep_reptile_v2.4.toml", [*overrides, "model.language_ranking.enabled=false"])
    assert not off.model.language_ranking.enabled and off.model.language_ranking.margin_mode == "relative"
    with pytest.raises(ValueError, match="language_ranking"):
        toml_config.load_train_config(TOML_DIR / "libero10_3ep_reptile_v2.4.toml", [*overrides, "model.language_ranking.foo=1"])
    with pytest.raises(ValueError, match="margin_mode"):
        toml_config.load_train_config(TOML_DIR / "libero10_3ep_reptile_v2.4.toml", [*overrides, "model.language_ranking.margin_mode=hinge"])
    reptile = (TOML_DIR / "reptile_pi05_v1.toml").read_text() + "\n[model.language_ranking]\nenabled = true\n"
    (tmp_path / "r.toml").write_text(reptile)
    with pytest.raises(ValueError, match="language_ranking"):
        toml_config.load_reptile_config(tmp_path / "r.toml")
