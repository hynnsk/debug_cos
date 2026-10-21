"""CPU end-to-end test of Reptile meta-training with the dummy pi05 model on a synthetic embodiment."""

import dataclasses
import os
import pathlib

import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

import openpi.models.pi0_config as pi0_config
import openpi.shared.normalize as normalize
import openpi.training.weight_loaders as weight_loaders
from openpi.hs11 import embodiments as _emb
from openpi.hs11 import reptile as _reptile
from openpi.hs11 import testing

from . import train_reptile


@pytest.fixture
def synthetic_embodiment(tmp_path, monkeypatch):
    root = testing.write_synthetic_v3_dataset(tmp_path / "data" / "synthetic", num_episodes=6, frames_per_episode=12, fps=10, size=24)
    spec = _emb.EmbodimentSpec(
        name="synthetic",
        root_subdir="synthetic",
        image_keys={"base": "observation.images.image"},
        state_keys=("observation.state",),
        action_keys=("action",),
        fps=10.0,
    )
    monkeypatch.setitem(_emb.EMBODIMENTS, "synthetic", spec)
    # Norm stats as compute_norm_stats_reptile.py would write them.
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    ds = __import__("openpi.hs11.lerobot_v3", fromlist=["x"]).LeRobotV3Dataset(spec.dataset_spec(str(tmp_path / "data")), 4, skip_images=True)
    stats["state"].update(np.stack([ds[i]["state"] for i in range(len(ds))]))
    stats["actions"].update(np.stack([ds[i]["actions"] for i in range(len(ds))]))
    normalize.save(tmp_path / "assets" / "synthetic", {k: v.get_statistics() for k, v in stats.items()})
    return tmp_path


def _config(tmp_path, **kwargs):
    return train_reptile.ReptileConfig(
        exp_name="test",
        name="reptile_test",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=4, discrete_state_input=False),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        embodiments=("synthetic",),
        data_root=str(tmp_path / "data"),
        assets_dir=str(tmp_path / "assets"),
        sampler=_reptile.SamplerConfig(k_shot=2, windows_per_demo=2, q_query=1, query_windows_per_demo=2),
        inner_steps=2,
        inner_batch_size=2,
        max_iter=3,
        save_interval=2,
        keep_period=1,
        eval_query_every=1,
        num_workers=kwargs.pop("num_workers", 0),
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        wandb_enabled=False,
        **kwargs,
    )


def test_reptile_trains_saves_and_resumes(synthetic_embodiment):
    import jax
    import openpi.models.model as _model

    tmp_path = synthetic_embodiment
    config = _config(tmp_path)
    train_reptile.main(config)
    ckpt = config.checkpoint_dir
    assert (ckpt / "2" / "params").is_dir() and (ckpt / "3" / "params").is_dir()
    log = [l for l in (ckpt / "reptile_train_log.jsonl").read_text().splitlines() if l.strip()]
    assert len(log) == 3
    import json

    rows = [json.loads(l) for l in log]
    assert rows[0]["embodiment"] == "synthetic" and rows[0]["query_loss_zero_shot"] is not None
    assert all(r["meta_step_norm"] > 0 for r in rows) and rows[0]["eps"] == pytest.approx(0.5)
    # the saved theta loads back as a plain params dict usable by CheckpointWeightLoader
    params = _model.restore_params(ckpt / "3" / "params", restore_type=np.ndarray)
    assert "PaliGemma" in params and "action_out_proj" in params
    # theta moved between the two checkpoints
    p2 = _model.restore_params(ckpt / "2" / "params", restore_type=np.ndarray)
    assert not np.allclose(p2["action_out_proj"]["kernel"], params["action_out_proj"]["kernel"])
    # resume from step 3 and run one more meta-iteration
    config2 = dataclasses.replace(config, resume=True, max_iter=4)
    train_reptile.main(config2)
    assert (ckpt / "4" / "params").is_dir()
    assert len([l for l in (ckpt / "reptile_train_log.jsonl").read_text().splitlines() if l.strip()]) == 4
    # downstream warm start: the weight loader merges the Reptile theta into a fresh model of the same config
    import flax.nnx as nnx

    loader = weight_loaders.CheckpointWeightLoader(str(ckpt / "4" / "params"))
    model_shape = nnx.eval_shape(config.model.create, jax.random.key(0))
    ref = nnx.state(model_shape).to_pure_dict()
    loaded = loader.load(ref)
    assert jax.tree.structure(loaded) == jax.tree.structure(ref)


def test_sampler_is_deterministic_and_disjoint():
    lengths = np.asarray([20, 20, 5, 20, 20, 20], dtype=np.int64)
    tasks = ("a", "a", "b", "b", "c", "c")
    table = _reptile.EmbodimentTable("e", lengths, tasks)
    cfg = _reptile.SamplerConfig(k_shot=2, windows_per_demo=3, q_query=1, min_frames=6)
    sampler = _reptile.EpisodicSampler({"e": table}, cfg, action_horizon=4, seed=1)
    a, b = sampler.sample(7), sampler.sample(7)
    assert a == b and len(a.support) == 6 and len(a.query) == 3
    assert 2 not in {e for e, _ in a.support} and not ({e for e, _ in a.support} & {e for e, _ in a.query})
    assert all(0 <= t <= 20 - 4 for _, t in a.support)
    assert sampler.sample(8) != a
    assert _reptile.meta_lr_at(0, 10, 0.5, 0.1) == pytest.approx(0.5) and _reptile.meta_lr_at(10, 10, 0.5, 0.1) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="usable episodes"):
        _reptile.EpisodicSampler({"e": table}, _reptile.SamplerConfig(k_shot=10, q_query=0), action_horizon=4, seed=1)


def test_reptile_runs_with_spawned_workers(synthetic_embodiment):
    config = _config(synthetic_embodiment, num_workers=2)
    config = dataclasses.replace(config, exp_name="workers", max_iter=2, save_interval=0, eval_query_every=1)
    train_reptile.main(config)
    rows = [l for l in (config.checkpoint_dir / "reptile_train_log.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == 2 and (config.checkpoint_dir / "2" / "params").is_dir()  # final checkpoint is always written
