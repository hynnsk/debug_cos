"""Synthetic LeRobot v3.0 datasets for CPU tests (parquet + real mp4 videos written with PyAV)."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_synthetic_v3_dataset(
    root: pathlib.Path,
    *,
    num_episodes: int = 4,
    frames_per_episode: int = 12,
    image_keys: tuple[str, ...] = ("observation.images.image",),
    state_dim: int = 8,
    action_dim: int = 7,
    fps: int = 10,
    size: int = 32,
    tasks: tuple[str, ...] = ("task a", "task b"),
    seed: int = 0,
) -> pathlib.Path:
    """Writes ``root`` with meta/info.json, meta/tasks.parquet, meta/episodes/..., data/..., videos/... (one file each)."""
    rng = np.random.default_rng(seed)
    root = pathlib.Path(root)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": None},
        "action": {"dtype": "float32", "shape": [action_dim], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for key in image_keys:
        features[key] = {"dtype": "video", "shape": [size, size, 3], "names": ["height", "width", "channel"], "info": {"video.fps": fps, "video.codec": "h264"}}
    info = {
        "codebase_version": "v3.0",
        "robot_type": "synthetic",
        "fps": fps,
        "total_episodes": num_episodes,
        "total_frames": num_episodes * frames_per_episode,
        "features": features,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    pq.write_table(pa.table({"task": list(tasks), "task_index": list(range(len(tasks)))}), root / "meta" / "tasks.parquet")

    rows, episodes, index = [], [], 0
    frames_by_key = {key: [] for key in image_keys}
    for ep in range(num_episodes):
        task = ep % len(tasks)
        episodes.append({"episode_index": ep, "length": frames_per_episode, "dataset_from_index": index, "tasks": str([tasks[task]])})
        for key in image_keys:
            episodes[-1][f"videos/{key}/chunk_index"] = 0
            episodes[-1][f"videos/{key}/file_index"] = 0
            episodes[-1][f"videos/{key}/from_timestamp"] = ep * frames_per_episode / fps
            episodes[-1][f"videos/{key}/to_timestamp"] = (ep + 1) * frames_per_episode / fps
        for t in range(frames_per_episode):
            rows.append(
                {
                    "observation.state": (rng.normal(size=state_dim) + ep).astype(np.float32).tolist(),
                    "action": (rng.normal(size=action_dim) * 0.1 + t).astype(np.float32).tolist(),
                    "timestamp": float(t / fps),
                    "frame_index": t,
                    "episode_index": ep,
                    "index": index,
                    "task_index": task,
                }
            )
            index += 1
            for key in image_keys:
                frame = np.full((size, size, 3), fill_value=(ep * 40 + t * 5) % 256, dtype=np.uint8)
                frame[:4, :4] = 255  # marker
                frames_by_key[key].append(frame)
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.Table.from_pylist(rows), root / "data" / "chunk-000" / "file-000.parquet")
    for key, frames in frames_by_key.items():
        path = root / "videos" / key / "chunk-000" / "file-000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_mp4(path, frames, fps)
    return root


def write_mp4(path: pathlib.Path, frames: list[np.ndarray], fps: int) -> None:
    import av

    with av.open(str(path), mode="w") as container:
        try:
            stream = container.add_stream("libx264", rate=fps)
            stream.pix_fmt = "yuv420p"
            stream.options = {"g": "4", "crf": "18"}  # short GOP keeps seeks cheap and exact
        except Exception:  # noqa: BLE001 -- no x264 in this PyAV build
            stream = container.add_stream("mpeg4", rate=fps)
            stream.pix_fmt = "yuv420p"
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        for frame in frames:
            packet = stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24"))
            container.mux(packet)
        container.mux(stream.encode())
