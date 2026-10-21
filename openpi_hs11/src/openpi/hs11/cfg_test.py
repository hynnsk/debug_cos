"""Classifier-free guidance in pi0.sample_actions (dummy pi05 on CPU) + prompt dropout knobs."""

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

import openpi.models.pi0_config as pi0_config
from openpi.hs11 import libero_v3
from openpi.models import tokenizer as _tokenizer


@pytest.fixture(scope="module")
def model_and_obs():
    config = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=4, discrete_state_input=False)
    model = config.create(jax.random.key(0))
    # pi05's adaRMSNorm modulation is zero-initialized (adaLN-zero), so a freshly initialized model ignores its prefix
    # (images + prompt) entirely. Randomize every zero-initialized kernel so the prompt influences the velocity field.
    import flax.nnx as nnx

    state = nnx.state(model)
    flat = state.flat_state()
    key = jax.random.key(42)
    for path, var in flat.items():
        arr = var.value
        if hasattr(arr, "ndim") and arr.ndim >= 2 and float(jnp.abs(arr).max()) == 0.0:
            key, sub = jax.random.split(key)
            var.value = 0.05 * jax.random.normal(sub, arr.shape, arr.dtype)
    nnx.update(model, nnx.State.from_flat_path(flat))
    obs, _ = config.inputs_spec(batch_size=2)
    rng = jax.random.key(1)
    obs = jax.tree.map(lambda spec: jax.random.uniform(rng, spec.shape, minval=-1, maxval=1) if spec.dtype == jnp.float32 else jnp.zeros(spec.shape, spec.dtype), obs)
    tok = _tokenizer.PaligemmaTokenizer(config.max_token_len)
    t, m = tok.tokenize("put the bowl on the plate")
    obs = obs.replace(tokenized_prompt=jnp.asarray(np.stack([t, t])), tokenized_prompt_mask=jnp.asarray(np.stack([m, m])))
    ut, um = tok.tokenize("")
    return model, obs, jnp.asarray(ut[None]), jnp.asarray(um[None])


def test_guidance_one_matches_plain_sampling_and_three_differs(model_and_obs):
    model, obs, ut, um = model_and_obs
    noise = jax.random.normal(jax.random.key(3), (2, model.action_horizon, model.action_dim))
    plain = model.sample_actions(jax.random.key(0), obs, num_steps=3, noise=noise)
    g1 = model.sample_actions(jax.random.key(0), obs, num_steps=3, noise=noise, guidance=1.0, uncond_tokenized_prompt=ut, uncond_tokenized_prompt_mask=um)
    g3 = model.sample_actions(jax.random.key(0), obs, num_steps=3, noise=noise, guidance=3.0, uncond_tokenized_prompt=ut, uncond_tokenized_prompt_mask=um)
    assert plain.shape == g1.shape == g3.shape == (2, model.action_horizon, model.action_dim)
    np.testing.assert_allclose(np.asarray(g1), np.asarray(plain), rtol=1e-4, atol=1e-4)  # guidance 1 == conditional branch
    assert not np.allclose(np.asarray(g3), np.asarray(plain), atol=1e-3)  # guidance 3 extrapolates away from the empty prompt
    # the unconditional branch alone (guidance 0) equals plain sampling with the empty prompt
    obs_u = obs.replace(tokenized_prompt=jnp.broadcast_to(ut, obs.tokenized_prompt.shape), tokenized_prompt_mask=jnp.broadcast_to(um, obs.tokenized_prompt_mask.shape))
    plain_u = model.sample_actions(jax.random.key(0), obs_u, num_steps=3, noise=noise)
    g0 = model.sample_actions(jax.random.key(0), obs, num_steps=3, noise=noise, guidance=0.0, uncond_tokenized_prompt=ut, uncond_tokenized_prompt_mask=um)
    np.testing.assert_allclose(np.asarray(g0), np.asarray(plain_u), rtol=1e-4, atol=1e-4)
    with pytest.raises(ValueError, match="mask"):
        model.sample_actions(jax.random.key(0), obs, num_steps=3, noise=noise, guidance=2.0, uncond_tokenized_prompt=ut)


def test_prompt_dropout_transform_and_toml_knob(tmp_path):
    import random

    random.seed(0)
    drop = libero_v3.PromptDropout(p=1.0)
    assert drop({"prompt": "x", "k": 1}) == {"prompt": "", "k": 1}
    assert libero_v3.PromptDropout(p=0.0)({"prompt": "x"}) == {"prompt": "x"}
    assert libero_v3.PromptDropout(p=1.0, uncond_prompt="do nothing")({"prompt": "x"})["prompt"] == "do nothing"
    assert libero_v3.PromptDropout(p=1.0)({"k": 1}) == {"k": 1}
    from openpi.hs11 import toml_config
    import openpi.training.config as _config

    root = __import__("pathlib").Path(__file__).resolve().parents[3]
    cfg = toml_config.load_train_config(root / "scripts/hs11/toml/goal_3ep_base_v1.toml", ["data.prompt_dropout=0.1"])
    assert cfg.data.prompt_dropout == 0.1
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert [type(t).__name__ for t in dc.repack_transforms.inputs] == ["RepackTransform", "PromptDropout"]
    assert _config.get_config("pi05_libero_v3_libero_goal_3ep_base").data.prompt_dropout == 0.0
    with pytest.raises(ValueError, match="prompt_dropout"):
        toml_config.load_train_config(root / "scripts/hs11/toml/goal_3ep_base_v1.toml", ["data.prompt_dropout=1.0"])
