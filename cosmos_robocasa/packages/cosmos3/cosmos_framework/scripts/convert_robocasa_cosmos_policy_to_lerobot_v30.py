# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Convert the Cosmos Policy RoboCasa export (``nvidia/RoboCasa-Cosmos-Policy``, ``success_only/*.hdf5``) to LeRobot v3.

cosmos_robocasa. One shard per task, ``<dst>/<Task>/<date>/lerobot`` (= the ``<root>/<task>/*/lerobot`` layout that
``RoboCasaLeRobotDataset`` discovers), with the SAME column contract as the RoboCasa365 LeRobot export so the reader works
unchanged (``use_base_action=False`` keeps the 10D arm block only):

    observation.images.robot0_{agentview_left,agentview_right,eye_in_hand}   224x224 video (AV1, 20 fps)
    action             12D [base_motion(4)=0, control_mode(1)=-1, eef_pos(3), eef_rot_axisangle(3), gripper(1)]
    observation.state  16D [base_pos(3)=0, base_quat_xyzw(4)=(0,0,0,1), eef_pos(3), eef_quat_xyzw(4), gripper_qpos(2)]
    task               the episode's ``task_description`` attribute

Source HDF5 (``data/demo_N``): ``actions`` = the 12D env action in robosuite order ``[arm_pos(3), arm_rot(3), gripper(1),
base(3), torso(1), mode(1)]`` -- Cosmos Policy trains on the first 7 (the base never moves in the 24 atomic tasks);
``robot_states`` = ``[gripper_qpos(2), eef_pos(3), eef_quat_xyzw(4)]`` (world frame); ``obs/*_rgb`` = (T, 224, 224, 3) uint8.
Demos are already no-op filtered and success-trimmed (Cosmos Policy ``regenerate_robocasa_dataset.py``).

Each shard also gets ``extras/stats_arrays.npz`` (the reader's 10D action ``[pos, rot6d, gripper]`` and 10D state token
``[eef_pos, rot6d(quat), finger opening]`` for every frame, built with the SAME rotation code as the reader) and
``extras/task_meta.json``. ``--stats-out`` aggregates the arrays of every shard into the bundled normalizer JSON
(blocks ``global`` / ``global_raw`` / ``state``, the LIBERO file layout) that ``action_normalization="quantile_rot"`` and
``state_normalization="minmax"`` read.

Usage (from packages/cosmos3 in the cosmos3-pt env; examples/setup_robocasa.sh wraps it)::

    PYTHONPATH=. python -m cosmos_framework.scripts.convert_robocasa_cosmos_policy_to_lerobot_v30 \\
        --src $COSMOS_STORAGE/data/robocasa_cosmos_policy/success_only \\
        --dst $COSMOS_STORAGE/data/robocasa_cosmos_policy_lerobot_v3 --jobs 6 \\
        --stats-out cosmos_framework/data/generator/action/normalizer_stats/robocasa_cosmos_policy_frame_wise_relative_rot6d.json
    ... --stats-only --stats-out <json>      # only aggregate the arrays of already converted shards
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

FPS = 20
IMAGE_HW = (224, 224)
CAMERAS = {
    # LeRobot feature name                       : HDF5 obs key
    "observation.images.robot0_agentview_left": "robot0_agentview_left_rgb",
    "observation.images.robot0_agentview_right": "robot0_agentview_right_rgb",
    "observation.images.robot0_eye_in_hand": "robot0_eye_in_hand_rgb",
}
ACTION_NAMES = [
    "base_forward", "base_side", "base_yaw", "torso_height",  # base_motion(4), zero (fixed-base tasks)
    "control_mode",  # -1 = arm mode
    "eef_pos_x", "eef_pos_y", "eef_pos_z",
    "eef_rot_ax", "eef_rot_ay", "eef_rot_az",
    "gripper",
]
STATE_NAMES = [
    "base_pos_x", "base_pos_y", "base_pos_z",
    "base_quat_x", "base_quat_y", "base_quat_z", "base_quat_w",
    "eef_pos_x", "eef_pos_y", "eef_pos_z",
    "eef_quat_x", "eef_quat_y", "eef_quat_z", "eef_quat_w",
    "gripper_qpos_0", "gripper_qpos_1",
]
ROTATION_DIMS = [3, 4, 5, 6, 7, 8]  # rot6d block of the reader's 10D [pos(3), rot6d(6), gripper(1)]
DEFAULT_TASKS = (
    "PnPCounterToCab", "PnPCabToCounter", "PnPCounterToSink", "PnPSinkToCounter", "PnPCounterToMicrowave",
    "PnPMicrowaveToCounter", "PnPCounterToStove", "PnPStoveToCounter", "OpenSingleDoor", "CloseSingleDoor",
    "OpenDoubleDoor", "CloseDoubleDoor", "OpenDrawer", "CloseDrawer", "TurnOnSinkFaucet", "TurnOffSinkFaucet",
    "TurnSinkSpout", "TurnOnStove", "TurnOffStove", "CoffeeSetupMug", "CoffeeServeMug", "CoffeePressButton",
    "TurnOnMicrowave", "TurnOffMicrowave",
)


def lerobot_features() -> dict:
    feats: dict = {}
    for key in CAMERAS:
        feats[key] = {"dtype": "video", "shape": (IMAGE_HW[0], IMAGE_HW[1], 3), "names": ["height", "width", "channel"]}
    feats["observation.state"] = {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES}
    feats["action"] = {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES}
    return feats


# ---------------------------------------------------------------------------------------------- pure array mappings
def build_action12(actions: np.ndarray) -> np.ndarray:
    """HDF5 ``actions`` (T, >=7) in robosuite order -> the RoboCasa365 12D layout the reader slices (5:8 / 8:11 / 11)."""
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] < 7:
        raise ValueError(f"actions must be (T, >=7), got {a.shape}")
    out = np.zeros((a.shape[0], 12), dtype=np.float32)
    out[:, 4] = -1.0  # control_mode: arm mode (RoboCasa365 convention; the base is never driven here)
    out[:, 5:8] = a[:, 0:3]  # eef position delta command
    out[:, 8:11] = a[:, 3:6]  # eef rotation delta command (axis-angle)
    out[:, 11] = a[:, 6]  # gripper (+1 close / -1 open)
    return out


def build_state16(robot_states: np.ndarray) -> np.ndarray:
    """HDF5 ``robot_states`` (T, 9) = [gripper_qpos(2), eef_pos(3), eef_quat_xyzw(4)] -> the 16D RoboCasa365 state."""
    rs = np.asarray(robot_states, dtype=np.float32)
    if rs.ndim != 2 or rs.shape[1] != 9:
        raise ValueError(f"robot_states must be (T, 9), got {rs.shape}")
    out = np.zeros((rs.shape[0], 16), dtype=np.float32)
    out[:, 6] = 1.0  # base_quat_xyzw = identity (the base pose is not recorded; the reader never uses it for the arm contract)
    out[:, 7:10] = rs[:, 2:5]  # eef_pos
    out[:, 10:14] = rs[:, 5:9]  # eef_quat (xyzw, robosuite convention)
    out[:, 14:16] = rs[:, 0:2]  # gripper_qpos (two fingers)
    return out


def reader_action10(actions: np.ndarray) -> np.ndarray:
    """What ``RoboCasaLeRobotDataset._build_frame_wise_action`` emits: [pos(3), rot6d(6), gripper(1)] (same rotation code)."""
    import torch

    from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation

    a = torch.from_numpy(np.asarray(actions, dtype=np.float32))
    matrix = convert_rotation(a[:, 3:6], input_format="axisangle", output_format="matrix")
    rot6d = convert_rotation(matrix, input_format="matrix", output_format="rot6d")
    return torch.cat([a[:, 0:3], rot6d, a[:, 6:7]], dim=-1).numpy().astype(np.float32)


def reader_state10(robot_states: np.ndarray) -> np.ndarray:
    """What ``RoboCasaLeRobotDataset._build_initial_state`` emits: [eef_pos(3), rot6d(6), qpos0 - qpos1]."""
    import torch

    from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation

    rs = torch.from_numpy(np.asarray(robot_states, dtype=np.float32))
    matrix = convert_rotation(rs[:, 5:9], input_format="quat_xyzw", output_format="matrix")
    rot6d = convert_rotation(matrix, input_format="matrix", output_format="rot6d")
    grip = rs[:, 0:1] - rs[:, 1:2]
    return torch.cat([rs[:, 2:5], rot6d, grip], dim=-1).numpy().astype(np.float32)


def _stats(a: np.ndarray) -> dict[str, list[float]]:
    finite = np.isfinite(a).all(axis=-1)
    a = a[finite].astype(np.float64)
    return {
        "mean": a.mean(axis=0).round(6).tolist(),
        "std": a.std(axis=0).round(6).tolist(),
        "min": a.min(axis=0).round(6).tolist(),
        "max": a.max(axis=0).round(6).tolist(),
        "q01": np.quantile(a, 0.01, axis=0).round(6).tolist(),
        "q99": np.quantile(a, 0.99, axis=0).round(6).tolist(),
    }


def compute_stats_blocks(action10: np.ndarray, state10: np.ndarray) -> dict:
    """``global_raw`` = raw stats of the 10D action (what ``quantile_rot`` reads); ``global`` = the same with the rot6d
    dims replaced by identity stats (what plain ``quantile`` / ``minmax`` read, as in the LIBERO file); ``state`` = the
    10D proprioception token (``state_normalization``)."""
    global_raw = _stats(action10)
    global_ = {k: list(v) for k, v in global_raw.items()}
    identity = {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0}
    for key, value in identity.items():
        for d in ROTATION_DIMS:
            global_[key][d] = value
    return {"global": global_, "global_raw": global_raw, "state": _stats(state10)}


# ---------------------------------------------------------------------------------------------- per-task conversion
def discover_sources(src: Path, tasks: tuple[str, ...]) -> list[tuple[str, str, Path]]:
    """``<src>/<category>/<Task>/<date>/*.hdf5`` -> [(task, date, path)] for the requested tasks (one file per task)."""
    found: list[tuple[str, str, Path]] = []
    for task in tasks:
        matches = sorted(glob.glob(str(src / "*" / task / "*" / "*.hdf5")))
        if not matches:
            print(f"WARNING: no HDF5 for task {task!r} under {src}", flush=True)
            continue
        if len(matches) > 1:
            print(f"WARNING: task {task!r} has {len(matches)} HDF5 files; using all of them as one shard per date", flush=True)
        for m in matches:
            found.append((task, Path(m).parent.name, Path(m)))
    return found


def convert_task(
    task: str,
    date: str,
    hdf5_path: str,
    dst_root: str,
    overwrite: bool = False,
    image_writer_threads: int = 4,
    vcodec: str = "libsvtav1",
) -> dict:
    """One HDF5 -> ``<dst_root>/<task>/<date>/lerobot`` (+ extras). Returns a summary dict. Safe to run in a subprocess."""
    import h5py
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(dst_root) / task / date / "lerobot"
    if (out / "meta" / "info.json").exists() and (out / "extras" / "stats_arrays.npz").exists() and not overwrite:
        meta = json.loads((out / "extras" / "task_meta.json").read_text())
        meta["skipped"] = True
        return meta
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ds = LeRobotDataset.create(
        repo_id=f"robocasa_cosmos_policy/{task}",
        fps=FPS,
        features=lerobot_features(),
        root=out,
        robot_type="PandaMobile",
        use_videos=True,
        image_writer_threads=image_writer_threads,
        vcodec=vcodec,
    )
    action10_parts, state10_parts, lengths, instructions = [], [], [], []
    with h5py.File(hdf5_path, "r") as f:
        data_attrs = {k: (v.tolist() if hasattr(v, "tolist") else str(v)) for k, v in f["data"].attrs.items()}
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for demo in demos:
            g = f["data"][demo]
            lang = str(g.attrs["task_description"])
            actions = g["actions"][:]  # (T, 12) float64 env actions
            robot_states = g["robot_states"][:]  # (T, 9)
            frames = {key: g["obs"][hkey][:] for key, hkey in CAMERAS.items()}  # (T, 224, 224, 3) uint8
            n = int(actions.shape[0])
            for key, arr in frames.items():
                if arr.shape != (n, IMAGE_HW[0], IMAGE_HW[1], 3) or arr.dtype != np.uint8:
                    raise ValueError(f"{hdf5_path}:{demo} {key}: unexpected images {arr.shape} {arr.dtype}")
            if robot_states.shape != (n, 9):
                raise ValueError(f"{hdf5_path}:{demo}: robot_states {robot_states.shape} vs actions {actions.shape}")
            action12 = build_action12(actions[:, :7])
            state16 = build_state16(robot_states)
            for t in range(n):
                frame = {key: np.ascontiguousarray(arr[t]) for key, arr in frames.items()}
                frame["observation.state"] = state16[t]
                frame["action"] = action12[t]
                frame["task"] = lang
                ds.add_frame(frame)
            ds.save_episode()
            action10_parts.append(reader_action10(actions[:, :7]))
            state10_parts.append(reader_state10(robot_states))
            lengths.append(n)
            instructions.append(lang)
    if getattr(ds, "image_writer", None) is not None:
        ds.stop_image_writer()
    ds.finalize()
    # lerobot leaves the (empty) PNG staging tree behind once every episode is encoded.
    shutil.rmtree(out / "images", ignore_errors=True)

    extras = out / "extras"
    extras.mkdir(exist_ok=True)
    action10 = np.concatenate(action10_parts, axis=0)
    state10 = np.concatenate(state10_parts, axis=0)
    np.savez(extras / "stats_arrays.npz", action10=action10, state10=state10, episode_lengths=np.asarray(lengths))
    meta = {
        "task": task,
        "date": date,
        "source": str(hdf5_path),
        "num_episodes": len(lengths),
        "num_frames": int(sum(lengths)),
        "episode_lengths": lengths,
        "instructions": sorted(set(instructions)),
        "fps": FPS,
        "vcodec": vcodec,
        "source_data_attrs": data_attrs,
        "converted_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
    }
    (extras / "task_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _convert_task_entry(args: tuple) -> dict:
    task, date, path, dst, overwrite, threads, vcodec = args
    try:
        return convert_task(task, date, path, dst, overwrite, threads, vcodec)
    except Exception as e:  # noqa: BLE001 -- report and keep the other tasks going
        return {"task": task, "date": date, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------------------------- stats aggregation
def aggregate_stats(dst_root: Path, tasks: tuple[str, ...], out_json: Path) -> dict:
    action_parts, state_parts, shards, num_episodes = [], [], [], 0
    for task in tasks:
        for npz in sorted(glob.glob(str(dst_root / task / "*" / "lerobot" / "extras" / "stats_arrays.npz"))):
            z = np.load(npz)
            action_parts.append(z["action10"])
            state_parts.append(z["state10"])
            num_episodes += int(len(z["episode_lengths"]))
            shards.append(str(Path(npz).parents[1].relative_to(dst_root)))
    if not action_parts:
        raise FileNotFoundError(f"no extras/stats_arrays.npz under {dst_root} for tasks {list(tasks)}; convert first")
    action10 = np.concatenate(action_parts, axis=0)
    state10 = np.concatenate(state_parts, axis=0)
    blocks = compute_stats_blocks(action10, state10)
    payload = {
        "metadata": {
            "embodiment_type": "robocasa",
            "dataset_name": "robocasa_cosmos_policy",
            "dataset_class": "RoboCasaLeRobotDataset",
            "source": "nvidia/RoboCasa-Cosmos-Policy success_only (24 atomic tasks, human demos, 224x224, no-op filtered, success trimmed)",
            "pose_convention": "frame_wise_relative",
            "pose_coordinate_frame": "native",
            "rotation_format": "6d",
            "action_dim": 10,
            "action_layout": "[eef_pos_delta(3), eef_rot6d_delta(6), gripper(1)] (robosuite OSC command units)",
            "skip_rotation_dims": ROTATION_DIMS,
            "state_dim": 10,
            "state_layout": "[eef_pos(3) world, eef_rot6d(6), gripper_opening(1) = qpos0 - qpos1] (the use_state token)",
            "chunk_length": 16,
            "num_frames": int(action10.shape[0]),
            "num_episodes": num_episodes,
            "num_shards": len(shards),
            "shards": shards,
            "generated": _dt.datetime.now().isoformat(timespec="seconds"),
            "generator": "cosmos_framework/scripts/convert_robocasa_cosmos_policy_to_lerobot_v30.py --stats-out",
        },
        **blocks,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    return payload


# ---------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="<hf local dir>/success_only (the kitchen_*/<Task>/<date>/*.hdf5 tree)")
    ap.add_argument("--dst", required=True, help="output root: <dst>/<Task>/<date>/lerobot (= ROBOCASA_ROOT)")
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of tasks (default: the 24 Cosmos Policy tasks)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel task conversions (processes)")
    ap.add_argument("--image-writer-threads", type=int, default=4, help="PNG writer threads per task process")
    ap.add_argument("--vcodec", default="libsvtav1", help="lerobot video codec (libsvtav1 = the LIBERO v3 AV1 files)")
    ap.add_argument("--overwrite", action="store_true", help="re-convert shards that already exist")
    ap.add_argument("--stats-out", default=None, help="write the aggregated normalizer JSON here after converting")
    ap.add_argument("--stats-only", action="store_true", help="skip conversion; only aggregate existing shards")
    args = ap.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    tasks = tuple(args.tasks) if args.tasks else DEFAULT_TASKS
    if not args.stats_only:
        sources = discover_sources(src, tasks)
        if not sources:
            print(f"ERROR: nothing to convert under {src}", file=sys.stderr)
            return 1
        dst.mkdir(parents=True, exist_ok=True)
        jobs = [(t, d, str(p), str(dst), args.overwrite, args.image_writer_threads, args.vcodec) for t, d, p in sources]
        print(f"converting {len(jobs)} shard(s) with {args.jobs} process(es) -> {dst}", flush=True)
        failures = 0
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=max(1, args.jobs), mp_context=mp.get_context("spawn")) as ex:
            futures = {ex.submit(_convert_task_entry, j): j for j in jobs}
            for fut in as_completed(futures):
                r = fut.result()
                if "error" in r:
                    failures += 1
                    print(f"FAILED {r['task']}/{r['date']}: {r['error']}\n{r['traceback']}", flush=True)
                else:
                    tag = "skipped (exists)" if r.get("skipped") else f"{r['seconds']} s"
                    print(
                        f"done {r['task']}/{r['date']}: {r['num_episodes']} episodes, {r['num_frames']} frames [{tag}] "
                        f"(elapsed {time.time() - t0:.0f} s)",
                        flush=True,
                    )
        if failures:
            print(f"ERROR: {failures} shard(s) failed", file=sys.stderr)
            return 1
    if args.stats_out:
        payload = aggregate_stats(dst, tasks, Path(args.stats_out))
        md = payload["metadata"]
        print(
            f"stats -> {args.stats_out}: {md['num_frames']} frames / {md['num_episodes']} episodes / {md['num_shards']} shards",
            flush=True,
        )
        for block in ("global_raw", "state"):
            print(f"  {block}.q01 = {np.round(payload[block]['q01'], 3).tolist()}")
            print(f"  {block}.q99 = {np.round(payload[block]['q99'], 3).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
