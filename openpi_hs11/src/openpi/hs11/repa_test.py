"""cosmos_hs11 DINOv2 REPA distillation in openpi: gemma hidden-state capture, the JAX DINOv2 against transformers, the two
loss terms, the pi0 wiring + gradients, checkpoint merging and the TOML knob (CPU, dummy pi05)."""

import dataclasses
import os
import pathlib

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.hs11 import dinov2_flax as _dino
from openpi.hs11 import repa as _repa
from openpi.hs11 import toml_config
from openpi.models import gemma as _gemma
from openpi.models import model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.weight_loaders as _weight_loaders

ROOT = pathlib.Path(__file__).resolve().parents[3]
TOML_DIR = ROOT / "scripts" / "hs11" / "toml"
REPA_TOMLS = ("libero10_3ep_reptile_v8.16", "goal_3ep_reptile_v8.9", "object_3ep_reptile_v8.12", "spatial_3ep_reptile_v8.12")
TINY = _dino.DINOv2Spec("dinov2_tiny_test", "", embed_dim=32, depth=2, num_heads=2)
HORIZON = 4
BATCH = 2


# ---------------------------------------------------------------------------------------------------------------
# gemma: capture of the first expert's residual stream after k blocks
# ---------------------------------------------------------------------------------------------------------------
def _gemma_inputs(cfg, b=2, tp=5, ts=3):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    prefix = jax.random.normal(k1, (b, tp, cfg.width))
    suffix = jax.random.normal(k2, (b, ts, cfg.width))
    cond = jax.random.normal(k3, (b, cfg.width))
    mask = jnp.ones((b, tp + ts, tp + ts), bool)
    positions = jnp.broadcast_to(jnp.arange(tp + ts)[None], (b, tp + ts))
    return prefix, suffix, positions, mask, cond


@pytest.fixture(scope="module")
def gemma_setup():
    cfg = _gemma.get_config("dummy")
    module = _gemma.Module(configs=[cfg, cfg], embed_dtype="float32", adarms=True)
    variables = nn.init(lambda m: m.init([False, True]), module)(jax.random.key(1))
    # RMSNorm scales are zero-initialized (1 + 0); perturb every parameter so each block visibly changes the stream.
    leaves, treedef = jax.tree.flatten(variables)
    keys = jax.random.split(jax.random.key(2), len(leaves))
    variables = jax.tree.unflatten(treedef, [x + 0.05 * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, keys, strict=True)])
    return cfg, module, variables, _gemma_inputs(cfg)


def test_capture_leaves_outputs_unchanged_and_last_layer_is_the_pre_norm_stream(gemma_setup):
    cfg, module, variables, (prefix, suffix, positions, mask, cond) = gemma_setup
    outs, kv = module.apply(variables, [prefix, suffix], positions, mask, [None, cond])
    outs_c, kv_c, captured = module.apply(
        variables, [prefix, suffix], positions, mask, [None, cond], capture_layer=cfg.depth, method="forward_capture"
    )
    for a, b in zip(outs, outs_c, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(kv[0]), np.asarray(kv_c[0]), rtol=1e-6, atol=1e-6)
    assert captured.shape == prefix.shape
    # the capture after the last block is the residual stream right before final_norm_0
    normed, _ = module.apply(variables, captured, None, method=lambda m, x, c: m.final_norms[0](x, c))
    np.testing.assert_allclose(np.asarray(normed), np.asarray(outs[0]), rtol=1e-5, atol=1e-5)
    assert float(jnp.abs(captured - prefix).max()) > 1e-3  # it is not the input embedding


@pytest.mark.parametrize("k", [1, 2, 3])
def test_capture_layer_k_equals_the_truncated_block_stack(gemma_setup, k):
    cfg, module, variables, (prefix, suffix, positions, mask, cond) = gemma_setup
    _, _, captured = module.apply(variables, [prefix, suffix], positions, mask, [None, cond], capture_layer=k, method="forward_capture")
    short_cfg = dataclasses.replace(cfg, depth=k)
    short = _gemma.Module(configs=[short_cfg, short_cfg], embed_dtype="float32", adarms=True)
    short_vars = {"params": dict(variables["params"])}
    short_vars["params"]["layers"] = jax.tree.map(lambda x: x[:k], variables["params"]["layers"])
    _, _, captured_short = short.apply(short_vars, [prefix, suffix], positions, mask, [None, cond], capture_layer=k, method="forward_capture")
    np.testing.assert_allclose(np.asarray(captured), np.asarray(captured_short), rtol=1e-5, atol=1e-5)
    if k > 1:
        _, _, previous = module.apply(variables, [prefix, suffix], positions, mask, [None, cond], capture_layer=k - 1, method="forward_capture")
        assert float(jnp.abs(captured - previous).max()) > 1e-4


def test_capture_is_differentiable_and_validated(gemma_setup):
    cfg, module, variables, (prefix, suffix, positions, mask, cond) = gemma_setup

    def f(p):
        _, _, captured = module.apply(variables, [p, suffix], positions, mask, [None, cond], capture_layer=2, method="forward_capture")
        return jnp.sum(captured**2)

    g = jax.grad(f)(prefix)
    assert np.isfinite(np.asarray(g)).all() and float(jnp.abs(g).max()) > 0
    with pytest.raises(ValueError, match="capture_layer"):
        module.apply(variables, [prefix, suffix], positions, mask, [None, cond], capture_layer=cfg.depth + 1, method="forward_capture")
    with pytest.raises(ValueError, match="first expert"):
        module.apply(variables, [None, suffix], positions[:, : suffix.shape[1]], mask[:, : suffix.shape[1], : suffix.shape[1]], [None, cond], capture_layer=1, method="forward_capture")


# ---------------------------------------------------------------------------------------------------------------
# DINOv2 in JAX
# ---------------------------------------------------------------------------------------------------------------
def _real_dinov2_path():
    try:
        return _dino.resolve_checkpoint(_dino.resolve_spec("dinov2_vitb14"))
    except Exception:  # noqa: BLE001  (no cache + no network)
        return None


@pytest.mark.parametrize("size", [224, 112], ids=["224_native_grid", "112_interpolated_posembed"])
def test_dinov2_jax_matches_transformers(size):
    path = _real_dinov2_path()
    if path is None:
        pytest.skip("facebook/dinov2-base snapshot not available")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    spec = _dino.resolve_spec("dinov2_vitb14")
    sd = _dino.load_hf_state(path)
    params = _dino.convert_hf_state(sd, spec, size, dtype=jnp.float32)
    img = np.asarray(jax.random.uniform(jax.random.key(3), (1, size, size, 3)), np.float32)
    ours = np.asarray(_dino.patch_tokens(params, spec, jnp.asarray(img)))

    hf = transformers.Dinov2Model.from_pretrained(str(path.parent), local_files_only=True).eval()
    x = (img - np.array(_dino.IMAGENET_MEAN, np.float32)) / np.array(_dino.IMAGENET_STD, np.float32)
    with torch.no_grad():
        ref = hf(pixel_values=torch.from_numpy(x).permute(0, 3, 1, 2)).last_hidden_state[:, 1:].numpy()
    assert ours.shape == ref.shape == (1, (size // 14) ** 2, 768)
    np.testing.assert_allclose(ours, ref, rtol=1e-3, atol=2e-3)


def test_dinov2_random_teacher_shapes_and_dtypes():
    params = _dino.random_params(TINY, jax.random.key(0), 56, dtype=jnp.bfloat16)
    tokens = _dino.patch_tokens(params, TINY, jnp.ones((3, 56, 56, 3)) * 0.5)
    assert tokens.shape == (3, 16, TINY.embed_dim) and tokens.dtype == jnp.float32
    with pytest.raises(ValueError, match="position embedding"):
        _dino.patch_tokens(params, TINY, jnp.ones((1, 112, 112, 3)))


# ---------------------------------------------------------------------------------------------------------------
# loss terms
# ---------------------------------------------------------------------------------------------------------------
def _numpy_losses(pred, target, mask, distance):
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    pn = pred / np.maximum(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-6)
    tn = target / np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-6)
    cos = (pn * tn).sum(-1)
    cos_mean = (cos * mask).sum(-1) / mask.sum(-1)
    rel = []
    for i in range(pred.shape[0]):
        keep = mask[i] > 0
        d = pn[i][keep] @ pn[i][keep].T - tn[i][keep] @ tn[i][keep].T
        rel.append(np.abs(d).mean() if distance == "l1" else (d**2).mean())
    return 1 - cos_mean, cos_mean, np.array(rel)


@pytest.mark.parametrize("distance", ["l1", "l2"])
def test_repa_losses_match_numpy_and_ignore_masked_tokens(distance):
    k1, k2 = jax.random.split(jax.random.key(0))
    pred = jax.random.normal(k1, (2, 6, 5))
    target = jax.random.normal(k2, (2, 6, 5))
    mask = jnp.asarray([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], jnp.float32)
    cos_loss, cos_mean, rel = _repa.repa_losses(pred, target, mask, relation_distance=distance)
    e_cos_loss, e_cos_mean, e_rel = _numpy_losses(pred, target, np.asarray(mask), distance)
    np.testing.assert_allclose(np.asarray(cos_loss), e_cos_loss, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(cos_mean), e_cos_mean, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(rel), e_rel, rtol=1e-5)
    # masked tokens do not matter at all
    pred2 = pred.at[0, 4:].set(100.0)
    cos2, _, rel2 = _repa.repa_losses(pred2, target, mask, relation_distance=distance)
    np.testing.assert_allclose(np.asarray(cos2), np.asarray(cos_loss), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(rel2), np.asarray(rel), rtol=1e-6)
    # identical token sets: both terms vanish; a global rotation of the student keeps the relation term at zero
    zero_cos, one, zero_rel = _repa.repa_losses(target, target, mask, relation_distance=distance)
    np.testing.assert_allclose(np.asarray(zero_cos), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(one), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(zero_rel), 0.0, atol=1e-6)
    q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(5), (5, 5)))
    _, _, rot_rel = _repa.repa_losses(target @ q, target, mask, relation_distance=distance)
    np.testing.assert_allclose(np.asarray(rot_rel), 0.0, atol=1e-5)
    with pytest.raises(ValueError, match="relation_distance"):
        _repa.repa_losses(pred, target, mask, relation_distance="cosine")


def test_resize_token_grid():
    tokens = jax.random.normal(jax.random.key(0), (2, 64, 3))
    assert _repa.resize_token_grid(tokens, (8, 8), (8, 8)) is tokens
    out = _repa.resize_token_grid(tokens, (8, 8), (4, 4))
    assert out.shape == (2, 16, 3)
    const = jnp.ones((1, 64, 3)) * 2.5
    np.testing.assert_allclose(np.asarray(_repa.resize_token_grid(const, (8, 8), (16, 16))), 2.5, rtol=1e-6)
    with pytest.raises(ValueError, match="grid"):
        _repa.resize_token_grid(tokens, (7, 7), (4, 4))


def test_teacher_fn_resizes_input_and_grid(monkeypatch):
    monkeypatch.setitem(_dino.TEACHERS, TINY.name, TINY)
    cfg = pi0_config.RepaConfig(enabled=True, loss_weight=1.0, teacher=TINY.name, teacher_input_size=112, teacher_dtype="float32")
    params = _dino.random_params(TINY, jax.random.key(0), 112)
    fn = _repa.make_teacher_fn(cfg, params, grid=(16, 16))
    tokens = fn(jnp.zeros((2, 224, 224, 3)))  # [-1, 1] images at openpi's 224 -> teacher at 112 (8x8) -> 16x16 grid
    assert tokens.shape == (2, 256, TINY.embed_dim) and np.isfinite(np.asarray(tokens)).all()


# ---------------------------------------------------------------------------------------------------------------
# pi0 wiring (dummy pi05 + tiny random teacher)
# ---------------------------------------------------------------------------------------------------------------
def _pi0_config(**repa) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=HORIZON,
        discrete_state_input=False,
        repa=pi0_config.RepaConfig(**repa),
    )


@pytest.fixture(scope="module")
def pi0_batch():
    config = _pi0_config()
    obs, _ = config.inputs_spec(batch_size=BATCH)
    rng = jax.random.key(1)
    obs = jax.tree.map(
        lambda spec: jax.random.uniform(rng, spec.shape, minval=-1, maxval=1) if spec.dtype == jnp.float32 else jnp.zeros(spec.shape, spec.dtype),
        obs,
    )
    masks = dict(obs.image_masks)
    masks["base_0_rgb"] = jnp.ones((BATCH,), bool)
    masks["left_wrist_0_rgb"] = jnp.ones((BATCH,), bool)
    masks["right_wrist_0_rgb"] = jnp.zeros((BATCH,), bool)  # LIBERO: empty slot
    obs = obs.replace(image_masks=masks, tokenized_prompt_mask=jnp.ones_like(obs.tokenized_prompt_mask))
    actions = jax.random.normal(jax.random.key(2), (BATCH, HORIZON, config.action_dim))
    return obs, actions


@pytest.fixture(scope="module")
def tiny_teacher():
    _dino.TEACHERS[TINY.name] = TINY  # module-scoped registration of the test-only architecture
    yield _repa.make_teacher_fn(
        pi0_config.RepaConfig(enabled=True, loss_weight=1.0, teacher=TINY.name, teacher_dtype="float32"),
        _dino.random_params(TINY, jax.random.key(0), 224),
    )
    _dino.TEACHERS.pop(TINY.name, None)


def test_pi0_repa_terms_add_up_and_are_masked(pi0_batch, tiny_teacher):
    obs, actions = pi0_batch
    cfg = _pi0_config(enabled=True, loss_weight=0.3, relation_loss_weight=0.7, relation_distance="l1", layer_index=2, teacher=TINY.name, teacher_dtype="float32")
    model = cfg.create(jax.random.key(0))
    assert "repa_proj" in nnx.state(model).to_pure_dict()
    rng = jax.random.key(7)
    chunked, metrics = model.compute_loss_and_metrics(rng, obs, actions, repa_teacher=tiny_teacher)
    assert set(metrics) == {"repa_loss", "repa_cos_sim", "repa_rel_loss", "repa_weighted_loss"}
    assert np.isfinite(np.asarray(chunked)).all()

    # closed form from the model's own pieces: plain FM loss + the two per-sample terms over base + wrist tokens only
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, obs, train=False)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
    u_t = noise - actions
    v, captured = model._predict_velocity(observation, x_t, time, capture_layer=2)  # noqa: SLF001
    plain = jnp.mean(jnp.square(v - u_t), axis=-1)
    names = list(observation.images)
    n_tok = 16 * 16
    pred = model.repa_proj(captured[:, : len(names) * n_tok])
    target = jnp.concatenate([tiny_teacher(observation.images[n]) for n in names], axis=1)
    mask = jnp.repeat(jnp.stack([observation.image_masks[n] for n in names], 1).astype(jnp.float32), n_tok, axis=1)
    cos_loss, cos_mean, rel = _repa.repa_losses(pred, target, mask, relation_distance="l1")
    expected = np.asarray(plain) + (0.3 * np.asarray(cos_loss) + 0.7 * np.asarray(rel))[:, None]
    np.testing.assert_allclose(np.asarray(chunked), expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(float(metrics["repa_loss"]), float(np.mean(np.asarray(cos_loss))), rtol=1e-4)
    np.testing.assert_allclose(float(metrics["repa_rel_loss"]), float(np.mean(np.asarray(rel))), rtol=1e-4)
    np.testing.assert_allclose(float(metrics["repa_cos_sim"]), float(np.mean(np.asarray(cos_mean))), rtol=1e-4)
    assert 0.0 < float(metrics["repa_loss"]) < 2.0 and float(metrics["repa_rel_loss"]) > 0.0

    # the masked right-wrist slot does not influence the loss, the wrist view does
    images = dict(obs.images)
    images["right_wrist_0_rgb"] = -images["right_wrist_0_rgb"]
    chunked_rw, _ = model.compute_loss_and_metrics(rng, obs.replace(images=images), actions, repa_teacher=tiny_teacher)
    np.testing.assert_allclose(np.asarray(chunked_rw), np.asarray(chunked), rtol=1e-5, atol=1e-6)


def test_pi0_repa_gradients_reach_projector_and_llm_and_disabled_model_has_no_projector(pi0_batch, tiny_teacher):
    obs, actions = pi0_batch
    cfg = _pi0_config(enabled=True, loss_weight=1.0, relation_loss_weight=1.0, layer_index=1, teacher=TINY.name, teacher_dtype="float32")
    model = cfg.create(jax.random.key(0))

    def loss_fn(m):
        chunked, _ = m.compute_loss_and_metrics(jax.random.key(3), obs, actions, repa_teacher=tiny_teacher)
        return jnp.mean(chunked)

    grads = nnx.grad(loss_fn)(model)
    flat = {"/".join(str(k) for k in path): v for path, v in grads.flat_state().items()}
    proj = [v.value for k, v in flat.items() if "repa_proj" in k]
    llm = [v.value for k, v in flat.items() if "/llm/" in k and "layers" in k]
    assert proj and llm
    assert all(np.isfinite(np.asarray(g)).all() for g in proj)
    assert max(float(jnp.abs(g).max()) for g in proj) > 0.0
    assert max(float(jnp.abs(g).max()) for g in llm) > 0.0

    with pytest.raises(ValueError, match="repa_teacher"):
        model.compute_loss(jax.random.key(0), obs, actions)

    plain = _pi0_config().create(jax.random.key(0))
    assert "repa_proj" not in nnx.state(plain).to_pure_dict()
    with pytest.raises(ValueError, match="layer_index"):
        _pi0_config(enabled=True, loss_weight=1.0, layer_index=99, teacher=TINY.name).create(jax.random.key(0))


def test_checkpoint_loader_keeps_the_fresh_projector():
    ref = {"PaliGemma": {"llm": {"w": jax.ShapeDtypeStruct((2,), jnp.float32)}}, "repa_proj": {"fc1": {"kernel": jax.ShapeDtypeStruct((2, 3), jnp.float32)}}}
    loaded = {"PaliGemma": {"llm": {"w": np.zeros((2,), np.float32)}}}
    merged = _weight_loaders._merge_params(loaded, ref, missing_regex=_weight_loaders.MISSING_FROM_CHECKPOINT_REGEX)  # noqa: SLF001
    assert isinstance(merged["PaliGemma"]["llm"]["w"], np.ndarray)
    assert isinstance(merged["repa_proj"]["fc1"]["kernel"], jax.ShapeDtypeStruct)  # stripped later -> fresh init survives


# ---------------------------------------------------------------------------------------------------------------
# config / TOML
# ---------------------------------------------------------------------------------------------------------------
def test_repa_config_validation():
    with pytest.raises(ValueError, match="loss_weight"):
        pi0_config.RepaConfig(loss_weight=-1.0)
    with pytest.raises(ValueError, match="relation_distance"):
        pi0_config.RepaConfig(relation_distance="cos")
    with pytest.raises(ValueError, match="layer_index"):
        pi0_config.RepaConfig(layer_index=0)
    with pytest.raises(ValueError, match="teacher_input_size"):
        pi0_config.RepaConfig(teacher_input_size=200)
    with pytest.raises(ValueError, match="projector_type"):
        pi0_config.RepaConfig(projector_type="conv")
    with pytest.raises(ValueError, match="both loss_weight and relation_loss_weight"):
        pi0_config.RepaConfig(enabled=True)
    with pytest.raises(ValueError, match="Unknown DINOv2 teacher"):
        _dino.resolve_spec("dinov3")
    assert _dino.resolve_spec("dinov2-base").embed_dim == 768


@pytest.mark.parametrize("stem", REPA_TOMLS)
def test_repa_tomls_enable_the_distillation_on_top_of_their_recipe(stem):
    path = TOML_DIR / f"{stem}.toml"
    cfg = toml_config.load_train_config(path, ["checkpoint.init=base"])  # the Reptile theta need not exist on this machine
    repa = cfg.model.repa
    assert repa.enabled and repa.loss_weight == 0.02 and repa.relation_loss_weight == 0.02 and repa.relation_distance == "l1"
    assert repa.layer_index == 6 and repa.teacher == "dinov2_vitb14" and repa.teacher_input_size == 224 and repa.teacher_checkpoint_path is None
    assert repa.projector_type == "mlp" and repa.projector_hidden_dim == 2048 and repa.teacher_dtype == "bfloat16"
    assert cfg.model.language_ranking.enabled  # the original recipe is untouched
    assert cfg.exp_name == stem and cfg.model.action_horizon == 10
    launcher = ROOT / "scripts" / "hs11" / f"launch_{stem}.sh"
    assert launcher.is_file() and f"{stem}.toml" in launcher.read_text()
    off = toml_config.load_train_config(path, ["checkpoint.init=base", "model.repa.enabled=false"])
    assert not off.model.repa.enabled
    with pytest.raises(ValueError, match="unknown keys"):
        toml_config.load_train_config(path, ["checkpoint.init=base", "model.repa.lambda=1"])


def test_reptile_toml_rejects_repa():
    with pytest.raises(ValueError, match="model.repa"):
        toml_config.load_reptile_config(TOML_DIR / "reptile_pi05_v1.toml", ["model.repa.enabled=true"])
