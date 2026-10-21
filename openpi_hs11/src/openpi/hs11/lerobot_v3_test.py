import json
import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from openpi.hs11 import lerobot_v3
from openpi.hs11 import testing


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    return testing.write_synthetic_v3_dataset(tmp_path_factory.mktemp("v3"), num_episodes=4, frames_per_episode=12, fps=10)


def _spec(root, **kwargs):
    return lerobot_v3.V3DatasetSpec(
        roots=(str(root),),
        image_keys={"base": "observation.images.image"},
        state_keys=("observation.state",),
        action_keys=("action",),
        name="synthetic",
        **kwargs,
    )


def test_reads_frames_chunks_and_prompts(synthetic_root):
    ds = lerobot_v3.LeRobotV3Dataset(_spec(synthetic_root), action_horizon=5)
    assert len(ds) == 48 and ds.num_episodes == 4 and ds.fps == 10
    s = ds[0]
    assert s["image/base"].shape == (32, 32, 3) and s["image/base"].dtype == np.uint8
    assert s["state"].shape == (8,) and s["actions"].shape == (5, 7) and s["prompt"] == "task a"
    assert np.allclose(np.round(s["actions"][:, 0]), np.arange(5), atol=0.5)  # chunk = consecutive rows
    # end-of-episode padding repeats the last action; frames are decoded at the right timestamps
    last = ds.get_frame(1, 11)
    assert np.allclose(last["actions"], last["actions"][:1]) and last["prompt"] == "task b"
    assert int(last["image/base"][10, 10, 0]) == (1 * 40 + 11 * 5) % 256
    mid = ds.get_frame(2, 3)
    assert int(mid["image/base"][10, 10, 0]) == (2 * 40 + 3 * 5) % 256
    assert ds.locate(ds.flat_index(2, 3)) == (2, 3)
    assert ds.episode_task_key(0) == "task a" and ds.episode_task_key(1) == "task b"


def test_spec_is_hashable_and_cached(synthetic_root):
    a = lerobot_v3.build_dataset(_spec(synthetic_root), 5)
    b = lerobot_v3.build_dataset(_spec(synthetic_root), 5)
    assert a is b and hash(_spec(synthetic_root)) == hash(_spec(synthetic_root))
    assert _spec(synthetic_root).image_key_map == {"base": "observation.images.image"}


def test_episode_subset_filters_and_validates(synthetic_root, tmp_path):
    good = tmp_path / "subset.json"
    good.write_text(json.dumps({"tasks": {"0": {"episodes": [0]}, "1": {"episodes": [3]}}}))
    ds = lerobot_v3.LeRobotV3Dataset(_spec(synthetic_root, episode_subset=str(good)), action_horizon=5)
    assert ds.num_episodes == 2 and len(ds) == 24 and {ds.get_frame(i, 0)["episode_index"] for i in range(2)} == {0, 3}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"tasks": {"0": {"episodes": [1]}}}))  # episode 1 belongs to task 1
    with pytest.raises(ValueError, match="maps it to task"):
        lerobot_v3.LeRobotV3Dataset(_spec(synthetic_root, episode_subset=str(bad)), action_horizon=5)
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"tasks": {"0": {"episodes": [99]}}}))
    with pytest.raises(ValueError, match="not in"):
        lerobot_v3.LeRobotV3Dataset(_spec(synthetic_root, episode_subset=str(missing)), action_horizon=5)


def test_skip_images_and_multi_root(synthetic_root, tmp_path):
    other = testing.write_synthetic_v3_dataset(tmp_path / "other", num_episodes=2, frames_per_episode=6, fps=10, tasks=("task c",), seed=1)
    spec = lerobot_v3.V3DatasetSpec(
        roots=(str(synthetic_root), str(other)),
        image_keys={"base": "observation.images.image"},
        state_keys=("observation.state",),
        action_keys=("action",),
        name="multi",
    )
    ds = lerobot_v3.LeRobotV3Dataset(spec, action_horizon=3, skip_images=True)
    assert ds.num_episodes == 6 and len(ds) == 48 + 12
    assert ds.get_frame(5, 0)["prompt"] == "task c" and ds.get_frame(5, 0)["image/base"].shape == (1, 1, 3)
    assert ds.episode_task_key(4) == "task c"


def test_dataset_pickles_without_tables_and_reloads(synthetic_root):
    import pickle

    ds = lerobot_v3.LeRobotV3Dataset(_spec(synthetic_root), action_horizon=5)
    _ = ds[3]  # load the tables
    clone = pickle.loads(pickle.dumps(ds))
    assert clone.roots[0]._frames is None and clone.roots[0]._lock is not None
    assert np.array_equal(clone[3]["actions"], ds[3]["actions"]) and clone[3]["prompt"] == ds[3]["prompt"]
