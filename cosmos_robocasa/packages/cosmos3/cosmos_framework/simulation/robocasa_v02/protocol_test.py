# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""cosmos_robocasa: the v0.2 closed-loop protocol helpers agree with the training reader (rot6d convention, state token,
state normalization, action decode round trip) and with Cosmos Policy's evaluation schedule; the server sends the
training-time view description; the eval scripts reference the right files."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from cosmos_framework.simulation.robocasa_v02 import protocol as P

ROOT = Path(__file__).resolve().parents[3]
STATS_JSON = ROOT / "cosmos_framework/data/generator/action/normalizer_stats/robocasa_cosmos_policy_frame_wise_relative_rot6d.json"


def test_schedule_matches_cosmos_policy() -> None:
    assert len(P.ROBOCASA_V02_TASKS) == 24 and set(P.TASK_MAX_STEPS) == set(P.ROBOCASA_V02_TASKS)
    assert P.TASK_MAX_STEPS["OpenDoubleDoor"] == 1000 and P.TASK_MAX_STEPS["CoffeePressButton"] == 300
    assert [P.scene_for_episode(e) for e in (0, 9, 10, 19, 20, 30, 40, 49, 50)] == [
        (1, 1), (1, 1), (2, 2), (2, 2), (4, 4), (6, 9), (7, 10), (7, 10), (1, 1)]
    assert P.env_seed_for_episode(195, 0) == 0 and P.env_seed_for_episode(195, 3) == 195 * 3 * 256
    assert P.env_seed_for_episode(195, 3, deterministic=False) is None
    kw = P.env_kwargs_for_episode("PnPCounterToCab", 12, 195, True)
    assert kw["robots"] == "PandaMobile" and kw["obj_instance_split"] == "B" and kw["layout_and_style_ids"] == ((2, 2),)
    assert kw["camera_widths"] == kw["camera_heights"] == 224 and kw["seed"] == 195 * 12 * 256
    assert kw["controller_configs"]["type"] == "OSC_POSE" and kw["controller_configs"]["output_max"][0] == 0.05
    assert "_comment" not in kw["controller_configs"] and kw["randomize_cameras"] is False
    with pytest.raises(ValueError):
        P.env_kwargs_for_episode("NavigateKitchen", 0, 195, True)


def test_rot6d_and_state_token_match_the_reader() -> None:
    from cosmos_framework.scripts.convert_robocasa_cosmos_policy_to_lerobot_v30 import reader_action10, reader_state10

    rng = np.random.default_rng(1)
    q = rng.normal(size=4).astype(np.float32); q /= np.linalg.norm(q)
    obs = {"robot0_eef_pos": rng.normal(size=3).astype(np.float32), "robot0_eef_quat": q,
           "robot0_gripper_qpos": np.array([0.03, -0.02], dtype=np.float32)}
    tok = P.raw_state_token(obs)
    rs9 = np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])[None]
    np.testing.assert_allclose(tok, reader_state10(rs9)[0], atol=1e-5)  # same rot6d convention as convert_rotation
    # action round trip: command axis-angle -> reader rot6d -> decode -> the same env command
    act7 = np.array([[0.3, -0.2, 0.1, 0.25, -0.4, 0.15, -1.0]], dtype=np.float32)
    a10 = reader_action10(act7)[0]
    env12 = P.decode_action10_to_env12(a10)
    np.testing.assert_allclose(env12[:7], act7[0], atol=1e-5)
    np.testing.assert_allclose(env12[7:], [0, 0, 0, 0, -1])
    assert P.decode_action10_to_env12(np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1.0]))[3:6].tolist() == [0, 0, 0]  # identity


def test_state_normalizer_matches_the_reader_normalizer(tmp_path: Path) -> None:
    from cosmos_framework.data.generator.action.utils.action_processing import resolve_action_normalization

    stats = {"state": {"min": list(np.linspace(-1, 1, 10)), "max": list(np.linspace(2, 4, 10)),
                       "q01": [0.0] * 10, "q99": [1.0] * 10, "mean": [0.5] * 10, "std": [2.0] * 10}}
    p = tmp_path / "s.json"; p.write_text(json.dumps(stats))
    tok = np.linspace(0.3, 2.9, 10).astype(np.float32)
    for method in ("minmax", "quantile", "meanstd"):
        ours = P.StateNormalizer(p, method)(tok)
        theirs = resolve_action_normalization(method, {k: torch.tensor(v) for k, v in stats["state"].items()}).normalize_action(torch.from_numpy(tok))
        np.testing.assert_allclose(ours, theirs.numpy(), atol=1e-6)
    with pytest.raises(KeyError):
        q = tmp_path / "nostate.json"; q.write_text(json.dumps({"global": stats["state"]})); P.StateNormalizer(q)


def test_bundled_state_stats_normalize_a_real_token_into_range() -> None:
    if not STATS_JSON.exists():
        pytest.skip("bundled stats not generated yet")
    norm = P.StateNormalizer(STATS_JSON, "minmax")
    st = json.loads(STATS_JSON.read_text())["state"]
    mid = (np.asarray(st["max"]) + np.asarray(st["min"])) / 2
    np.testing.assert_allclose(norm(mid), np.zeros(10), atol=1e-6)
    assert np.all(np.abs(norm(np.asarray(st["q99"]))) <= 1.0 + 1e-6)


def test_compose_left_wrist_is_upright_concat() -> None:
    left = np.zeros((224, 224, 3), np.uint8); left[0, :, 0] = 255  # top row red in the raw (bottom-up) render
    wrist = np.full((224, 224, 3), 7, np.uint8)
    comp = P.compose_left_wrist({"robot0_agentview_left_image": left, "robot0_eye_in_hand_image": wrist})
    assert comp.shape == (224, 448, 3) and comp[-1, 0, 0] == 255 and comp[0, 0, 0] == 0  # flipped upright
    assert (comp[:, 224:] == 7).all()
    tiled = P.compose_wrist_lr({"robot0_agentview_left_image": left, "robot0_eye_in_hand_image": wrist,
                                "robot0_agentview_right_image": wrist})
    assert tiled.shape == (336, 224, 3)


def test_server_view_description_follows_the_camera_set() -> None:
    from cosmos_framework.scripts.action_policy_server_robocasa import view_description_for_camera_set

    assert view_description_for_camera_set("left_wrist").startswith("The left half is a third-person view")
    assert "top row" in view_description_for_camera_set("wrist_lr")
    assert view_description_for_camera_set(None) is None and view_description_for_camera_set("lrw") is None
    src = (ROOT / "cosmos_framework/scripts/action_policy_server_robocasa.py").read_text()
    assert 'data_dict["additional_view_description"] = self.view_description' in src
    assert "view_description: str | None = None" in src


def test_eval_scripts_reference_the_v02_pipeline() -> None:
    ev = (ROOT / "examples/eval_robocasa_closed_loop.sh").read_text()
    for s in ("robocasa_v02/closed_loop_eval.py", "--action-stats-path", "--action-normalization quantile_rot",
              "--state-stats-path", "--state-normalization minmax", "--image-size 224", 'ACTION_HORIZON="${ACTION_HORIZON:-16}"',
              'NUM_TRIALS="${NUM_TRIALS:-50}"', 'SEED="${SEED:-195}"', "robocasa-v02-eval", "summarize_results.py",
              'export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"'):
        assert s in ev, s
    tasks = re.search(r'TASKS="\$\{TASKS:-([^}]*)\}"', ev).group(1).split()
    assert tuple(tasks) == P.ROBOCASA_V02_TASKS
    setup = (ROOT / "examples/setup_robocasa_v02_eval.sh").read_text()
    for s in ("moojink/robocasa-cosmos-policy", "robosuite==1.5.1", "mujoco==3.2.6", "download_kitchen_assets.py", "setup_macros.py",
              "Tej3/robocasa-v02-kitchen-assets", "sha256sum -c SHA256SUMS", 'ASSET_SOURCE="${ASSET_SOURCE:-auto}"'):
        assert s in setup, s
    assert "aigen" not in setup.split("for d in")[1].split("done")[0]  # the eval never needs aigen_objs
    assert (ROOT / "docs/collaborator_setup.md").is_file()
