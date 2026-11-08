# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Train/eval consistency check for the RoboCasa365 closed-loop pipeline -- cosmos_hs11 (robocasa-eval env, CPU ok).

Restores recorded demos (LeRobot v2.1 ``extras/episode_XXXXXX``: ep_meta.json + model.xml.gz + states.npz) in the SAME
environment the closed-loop eval builds (``closed_loop_eval.make_env``, Atomic-Seen protocol) and checks, per demo:

  1. the eval-side first observation (``compose``) against frame 0 of the dataset videos (mean |pixel diff| per view),
  2. the eval-side state token (``build_state_token``) against ``observation.state[0]`` re-encoded as in training,
  3. the language prompt the eval reads (``env.get_ep_meta()["lang"]``) against the dataset annotation,
  4. a replay of the recorded 12D actions through ``env.step`` -- once reordered to the env order and once through the
     reader 15D arm-first encoding + ``decode_15d_armfirst_to_env12`` (exactly what the eval loop does) -- reporting
     eef / gripper drift against the recorded states and whether ``_check_success`` fires.

2026-11-08 result (6 demos, 3 tasks, osmesa): pixel diff 1.6-4.0 / 255 (19 with translucent_robot=True, so the videos
are opaque), token error 0, prompts identical, replay drift ~1 mm with success at the recorded step in 5/6 (the 6th,
a knife-edge CloseBlenderLid placement, drifted 2.5 cm) => no observation / action / controller mismatch; the closed-loop
floor is the policy. Dataset frames must be extracted first (cosmos3-pt env, PyAV) -- see examples/eval_robocasa_consistency.sh.

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=. <robocasa-eval python> \
        cosmos_framework/simulation/robocasa/replay_consistency_check.py <ROBOCASA_V21_ROOT> <out_dir> \
        '{"TurnOnMicrowave": [0, 100], "PickPlaceCounterToCabinet": [0, 100]}'
"""

import glob
import gzip
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import numpy as np
import pandas as pd
import robocasa  # noqa: F401
from scipy.spatial.transform import Rotation as R

from cosmos_framework.simulation.robocasa import closed_loop_eval as cle

cle.CAMERA_SET = "left_wrist"
cle.USE_BASE_ACTION = True
cle.BASE_ENCODING = "raw_armfirst"
from cosmos_framework.simulation.robocasa.closed_loop_eval import (
    ATOMIC_SEEN_LAYOUT_STYLE,
    ATOMIC_SEEN_OBJ_SPLIT,
    build_state_token,
    compose,
    make_env,
)
from cosmos_framework.simulation.robocasa.eval_utils import decode_15d_armfirst_to_env12


def token_from_state(s0):
    m = R.from_quat(s0[10:14]).as_matrix()
    return np.concatenate([s0[7:10], m[:, 0], m[:, 1], [s0[14] - s0[15]]])


root, out = sys.argv[1], sys.argv[2]
plan = json.loads(sys.argv[3])


def load_ep(ds, ep):
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{ds}/data/**/*.parquet", recursive=True)))
    e = df[df.episode_index == ep].sort_values("frame_index")
    act = np.stack(e["action"].to_numpy()).astype(np.float64)
    st = np.stack(e["observation.state"].to_numpy()).astype(np.float64)
    ex = f"{ds}/extras/episode_{ep:06d}"
    meta = json.load(open(f"{ex}/ep_meta.json"))
    xml = gzip.open(f"{ex}/model.xml.gz").read().decode()
    states = np.load(f"{ex}/states.npz")["states"]
    return act, st, meta, xml, states


def restore(env, xml, state0, meta):
    if hasattr(env, "set_ep_meta"):
        env.set_ep_meta(meta)
    else:
        env._ep_meta = meta
    env.reset()
    xml = env.edit_model_xml(xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(state0)
    env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()
    return env._get_observations(force_update=True)


def to_env_order(a):  # dataset [base4, mode, pos3, rot3, grip] -> env [pos3, rot3, grip, base4, mode]
    return np.concatenate([a[5:8], a[8:11], a[11:12], a[0:4], a[4:5]])


def via_eval_decoder(a):  # reader 15D arm-first [pos3, rot6d, grip, base4, mode] -> decode_15d_armfirst_to_env12
    m = R.from_rotvec(a[8:11]).as_matrix()
    rot6d = np.concatenate([m[:, 0], m[:, 1]])
    a15 = np.concatenate([a[5:8], rot6d, a[11:12], a[0:4], a[4:5]]).astype(np.float32)
    return decode_15d_armfirst_to_env12(a15, False)


results = []
for task, eps in plan.items():
    ds = glob.glob(f"{root}/{task}/*/lerobot")[0]
    env = make_env(ds, 256, obj_split=ATOMIC_SEEN_OBJ_SPLIT, layout_style=ATOMIC_SEEN_LAYOUT_STYLE, seed=0)
    for ep in eps:
        act, st, meta, xml, states = load_ep(ds, ep)
        # 1-3. first frame / state token / prompt (eval side) vs dataset
        obs = restore(env, xml, states[0], meta)
        comp = compose(obs).astype(np.uint8)
        diffs = {}
        for cam, sl in (("robot0_agentview_left", slice(0, 256)), ("robot0_eye_in_hand", slice(256, 512))):
            f = f"{out}/{task}_ep{ep}_{cam}.npy"
            diffs[cam] = (
                round(float(np.abs(comp[:, sl].astype(np.float32) - np.load(f)[0].astype(np.float32)).mean()), 2)
                if os.path.exists(f)
                else None
            )
        tok = np.asarray(build_state_token(obs))[:10]
        tok_err = float(np.abs(tok - token_from_state(st[0])).max())
        print(
            json.dumps(
                dict(
                    task=task,
                    ep=ep,
                    pix_diff=diffs,
                    state_token_err=round(tok_err, 6),
                    lang_env=env.get_ep_meta().get("lang", ""),
                    lang_data=meta["lang"],
                )
            ),
            flush=True,
        )
        for label, fn in (("env_order", to_env_order), ("eval_decoder", via_eval_decoder)):
            restore(env, xml, states[0], meta)
            T = len(act)
            eef_err, grip_err, first_succ = [], [], None
            for t in range(T - 1):
                o, _, _, _ = env.step(fn(act[t]))
                eef_err.append(float(np.abs(np.asarray(o["robot0_base_to_eef_pos"]) - st[t + 1][7:10]).max()))
                grip_err.append(float(np.abs(np.asarray(o["robot0_gripper_qpos"]) - st[t + 1][14:16]).max()))
                if first_succ is None and env._check_success():
                    first_succ = t + 1
            r = dict(
                task=task,
                ep=ep,
                order=label,
                frames=T,
                replay_success_step=first_succ,
                final_success=bool(env._check_success()),
                eef_drift_mean=round(float(np.mean(eef_err)), 4),
                eef_drift_max=round(max(eef_err), 4),
                eef_drift_last=round(eef_err[-1], 4),
                grip_err_mean=round(float(np.mean(grip_err)), 4),
            )
            print(json.dumps(r), flush=True)
            results.append(r)
    env.close()
json.dump(results, open(f"{out}/replay_results2.json", "w"), indent=1)
print("done")
