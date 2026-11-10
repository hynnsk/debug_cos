# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Closed-loop evaluation of a Cosmos3-Edge RoboCasa checkpoint on the ORIGINAL RoboCasa benchmark (v0.2 simulator),
following Cosmos Policy's protocol (see ``protocol.py``). One task per invocation; the policy is served by
``cosmos_framework.scripts.action_policy_server_robocasa`` (cosmos3-pt env, GPU) and this client runs in the simulator env
(``examples/setup_robocasa_v02_eval.sh`` -> conda env robocasa-v02-eval: robosuite 1.5.1 + the Cosmos Policy fork of
robocasa v0.2 + its kitchen assets). Only pure-python modules of ``cosmos_framework`` are imported here, so
``PYTHONPATH=<repo>`` is enough.

    MUJOCO_GL=egl PYTHONPATH=. python cosmos_framework/simulation/robocasa_v02/closed_loop_eval.py \\
        --server-url http://127.0.0.1:8900 --task PnPCounterToCab --output-dir <out>/PnPCounterToCab \\
        --num-trials 50 --seed 195 --action-horizon 16 \\
        --state-stats-path cosmos_framework/data/generator/action/normalizer_stats/robocasa_cosmos_policy_frame_wise_relative_rot6d.json

The contract flags (``--action-horizon``, ``--camera-set``, ``--state-normalization``) MUST match the training recipe
(action_policy_robocasa_edge: 16 / left_wrist / minmax); the server must run with the same stats JSON
(``--action-stats-path`` + ``--action-normalization quantile_rot``) so the 10D actions come back denormalized.
Writes ``results.json`` (one record per rollout) + ``rolloutNN.mp4`` / ``rolloutNN_init.png`` into ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from cosmos_framework.simulation.robocasa.eval_utils import decode_pred_video, predict
from cosmos_framework.simulation.robocasa_v02.protocol import (
    COMPOSERS,
    DEFAULT_SEED,
    NUM_SETTLE_STEPS,
    ROBOCASA_V02_TASKS,
    TASK_MAX_STEPS,
    StateNormalizer,
    decode_action10_to_env12,
    env_kwargs_for_episode,
    env_seed_for_episode,
    raw_state_token,
    scene_for_episode,
)


def registry_horizon(task: str) -> int | None:
    """Horizon from the installed robocasa registry (cross-check of the bundled table); None if unavailable."""
    try:
        from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS

        return int(SINGLE_STAGE_TASK_DATASETS[task]["horizon"])
    except Exception:  # noqa: BLE001
        return None


def wait_for_server(server_url: str, timeout: float) -> dict:
    import requests

    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{server_url}/info", timeout=5)
            if r.ok:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(3)
    raise RuntimeError(f"server not ready at {server_url}")


def run_episode(env, *, task: str, prompt: str, compose, state_norm, server_url: str, action_horizon: int,
                max_steps: int, timeout: float, save_png: Path | None, save_video: Path | None,
                gen_video_path: Path | None, image_size: int) -> tuple[bool, int]:
    # Cosmos Policy: 10 zero-action steps so the scene settles before the first observation.
    obs = None
    for _ in range(NUM_SETTLE_STEPS):
        obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))
    first = compose(obs)
    if save_png is not None:
        from PIL import Image

        Image.fromarray(first).save(save_png)
    frames, gen_frames, queue = [first], [], []
    success, done_steps = False, max_steps
    for t in range(max_steps):
        if not queue:
            comp = compose(obs)
            token = state_norm(raw_state_token(obs)).tolist()
            result = predict(server_url, comp, prompt, image_size, timeout, state=token)
            if gen_video_path is not None and result.get("video"):
                gen_frames.extend(decode_pred_video(result["video"]))
            acts = result["action"]
            if not acts:
                raise RuntimeError("server returned no actions (see server.log)")
            queue = [np.asarray(a, dtype=np.float32) for a in acts[:action_horizon]]
        env_action = decode_action10_to_env12(queue.pop(0))
        obs, _, _, _ = env.step(env_action)
        frames.append(compose(obs))
        if bool(env._check_success()):
            success, done_steps = True, t + 1
            break
    if save_video is not None:
        import imageio.v2 as imageio

        imageio.mimwrite(str(save_video), frames, fps=20, macro_block_size=None)
    if gen_video_path is not None and gen_frames:
        import imageio.v2 as imageio

        imageio.mimwrite(str(gen_video_path), gen_frames, fps=20, macro_block_size=None)
    return success, done_steps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-url", default="http://127.0.0.1:8900")
    ap.add_argument("--task", required=True, choices=ROBOCASA_V02_TASKS)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-trials", type=int, default=50, help="rollouts per task (Cosmos Policy: 50)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="base seed (Cosmos Policy: 195 / 196 / 197)")
    ap.add_argument("--no-deterministic", action="store_true", help="unseeded envs (Cosmos Policy --deterministic False)")
    ap.add_argument("--action-horizon", type=int, default=16, help="actions executed per query (= chunk_length 16)")
    ap.add_argument("--camera-set", default="left_wrist", choices=sorted(COMPOSERS))
    ap.add_argument("--image-size", type=int, default=224, help="height sent to the server (224 = native, no rescale)")
    ap.add_argument("--state-stats-path", required=True, help="bundled stats JSON with the 'state' block")
    ap.add_argument("--state-normalization", default="minmax", choices=("minmax", "quantile", "meanstd"))
    ap.add_argument("--max-steps", type=int, default=None, help="override the registry horizon")
    ap.add_argument("--save-gen-video", action="store_true", help="also save the model's generated video per rollout")
    ap.add_argument("--no-video", action="store_true", help="skip the rollout mp4s (keep results.json + init pngs)")
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args(argv)

    import robosuite
    import robocasa  # noqa: F401  (registers the v0.2 kitchen envs)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    compose = COMPOSERS[args.camera_set]
    state_norm = StateNormalizer(args.state_stats_path, args.state_normalization)
    deterministic = not args.no_deterministic
    max_steps = args.max_steps or TASK_MAX_STEPS[args.task]
    reg = registry_horizon(args.task)
    if reg is not None and reg != TASK_MAX_STEPS[args.task]:
        print(f"[eval] WARNING: registry horizon {reg} != bundled {TASK_MAX_STEPS[args.task]} for {args.task}", flush=True)
    info = wait_for_server(args.server_url, args.timeout)
    print(f"[eval] server: {json.dumps({k: info.get(k) for k in ('run_name', 'checkpoint', 'action_chunk_size', 'raw_action_dim', 'requires_state', 'action_normalization', 'view_description', 'guidance', 'num_steps')})}", flush=True)
    if not info.get("requires_state"):
        print("[eval] WARNING: the server reports requires_state=False; the Cosmos3 RoboCasa recipe trains with a state token", flush=True)
    if info.get("action_normalization") not in (None, "quantile_rot"):
        print(f"[eval] WARNING: server action_normalization={info.get('action_normalization')!r} (recipe: quantile_rot)", flush=True)
    print(f"[eval] {args.task}: {args.num_trials} rollouts, seed {args.seed}, deterministic={deterministic}, horizon {max_steps}", flush=True)

    results: list[dict] = []
    t_task = time.time()
    for ep in range(args.num_trials):
        kw = env_kwargs_for_episode(args.task, ep, args.seed, deterministic)
        env = robosuite.make(**kw)
        env.reset()  # ONE reset per episode: every reset re-samples the scene / task instance
        prompt = env.get_ep_meta().get("lang", "") or ""
        t0 = time.time()
        ok, steps = run_episode(
            env, task=args.task, prompt=prompt, compose=compose, state_norm=state_norm, server_url=args.server_url,
            action_horizon=args.action_horizon, max_steps=max_steps, timeout=args.timeout,
            save_png=out / f"rollout{ep:02d}_init.png",
            save_video=None if args.no_video else out / f"rollout{ep:02d}.mp4",
            gen_video_path=(out / f"rollout{ep:02d}_generated.mp4") if args.save_gen_video else None,
            image_size=args.image_size,
        )
        env.close()
        rec = {"ep": ep, "policy": ok, "steps": steps, "prompt": prompt, "scene": list(scene_for_episode(ep)),
               "env_seed": env_seed_for_episode(args.seed, ep, deterministic), "seconds": round(time.time() - t0, 1)}
        results.append(rec)
        print(f"[eval] rollout {ep:02d} success={ok} steps={steps} scene={rec['scene']} prompt={prompt!r}", flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=2))  # progressive, so a crash keeps partial results
    n_ok = sum(1 for r in results if r["policy"])
    print(f"\n{args.task}: {n_ok}/{len(results)} successful rollouts ({time.time() - t_task:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
