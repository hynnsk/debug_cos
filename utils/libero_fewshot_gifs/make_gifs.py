#!/usr/bin/env python
"""Export the few-shot LIBERO demonstration episodes of one suite (episode-subset JSON) as GIFs.

    --suite libero_10 | libero_goal | libero_object | libero_spatial   (LeRobot v3 dirs under $COSMOS_STORAGE/data/LIBERO_LeRobot_v3)
picks the dataset root, the bundled subset JSON `<suite>_3ep_per_task_seed42.json` (newest cosmos_hs* repo) and the output
folder `project/utils/<suite>_3ep_seed42_gifs/`; each can be overridden (--libero-root / --episode-subset / --out-dir).

Output layout (one folder per task, named by its text instruction, one GIF per episode):
    <out-dir>/<text instruction>/ep{episode_index:03d}.gif      e.g. ".../turn on the stove and put the moka pot on it/ep038.gif"
    <out-dir>/manifest.json                                       what was written (task -> episodes -> gif, frames, size)

Default = both camera views side by side (3rd-person | wrist), native 256x256 per view, native 20 fps, every frame.
The AV1 mp4s are decoded with PyAV (libdav1d) exactly like utils/dinov2_pca_viz; GIFs are encoded by ffmpeg
(palettegen/paletteuse, one palette per GIF) from a raw RGB pipe. CPU only; the login node is enough.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent  # <...>/project

IMAGE_KEY = "observation.images.image"  # third-person camera
WRIST_KEY = "observation.images.wrist_image"  # wrist camera
VIEWS = {"image": (IMAGE_KEY,), "wrist": (WRIST_KEY,), "both": (IMAGE_KEY, WRIST_KEY)}
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
SUBSET_JSON_TEMPLATE = "{suite}_3ep_per_task_seed42.json"
OUT_DIR_TEMPLATE = "{suite}_3ep_seed42_gifs"  # under project/utils


@dataclass
class Episode:
    index: int
    task: str
    length: int
    videos: dict[str, tuple[str, float, float]] = field(default_factory=dict)  # view key -> (mp4, from_ts, to_ts)


def load_libero_meta(root: Path) -> tuple[float, list[Episode]]:
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {root / 'meta' / 'episodes'}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    episodes: list[Episode] = []
    for _, row in df.iterrows():
        ep = Episode(index=int(row["episode_index"]), task=str(list(row["tasks"])[0]), length=int(row["length"]))
        for key in (IMAGE_KEY, WRIST_KEY):
            rel = info["video_path"].format(
                video_key=key,
                chunk_index=int(row[f"videos/{key}/chunk_index"]),
                file_index=int(row[f"videos/{key}/file_index"]),
            )
            ep.videos[key] = (
                str(root / rel),
                float(row[f"videos/{key}/from_timestamp"]),
                float(row[f"videos/{key}/to_timestamp"]),
            )
        episodes.append(ep)
    episodes.sort(key=lambda e: e.index)
    return fps, episodes


def find_episode_subset_json(name_or_path: str) -> Path | None:
    """An explicit path, or the bundled file of the newest cosmos_hs* repo."""
    p = Path(name_or_path).expanduser()
    if p.is_file():
        return p
    pattern = str(PROJECT_ROOT / "cosmos_hs*" / "packages" / "cosmos3" / "cosmos_framework" / "data" / "generator"
                  / "action" / "episode_subsets" / name_or_path)
    hits = sorted(glob.glob(pattern), reverse=True)
    return Path(hits[0]) if hits else None


def decode_range_pyav(path: str, from_ts: float, to_ts: float, fps: float) -> np.ndarray:
    """All frames with from_ts <= t < to_ts of the concatenated mp4 -> [T,H,W,3] uint8."""
    import av

    tol = 0.5 / fps
    out: list[np.ndarray] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = stream.time_base
        container.seek(max(0, int((from_ts - 1.0) / tb)), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            ts = float(frame.pts * tb)
            if ts < from_ts - tol:
                continue
            if ts >= to_ts - tol:
                break
            out.append(frame.to_ndarray(format="rgb24"))
    if not out:
        raise RuntimeError(f"{path}: no frames decoded in [{from_ts:.2f}, {to_ts:.2f})")
    return np.stack(out, axis=0)


def write_gif_ffmpeg(frames: np.ndarray, out_path: Path, fps: float, scale: float) -> None:
    """frames [T,H,W,3] uint8 -> GIF via ffmpeg palettegen/paletteuse (single shared palette, bayer dither)."""
    t, h, w, _ = frames.shape
    vf = []
    if scale != 1.0:
        vf.append(f"scale=iw*{scale}:ih*{scale}:flags=lanczos")
    pre = ",".join(vf) + "," if vf else ""
    filt = f"[0:v]{pre}split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "pipe:0",
        "-filter_complex", filt, "-loop", "0", str(out_path),
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}:\n{proc.stderr.decode(errors='replace')}")


def write_gif_pillow(frames: np.ndarray, out_path: Path, fps: float, scale: float) -> None:
    """Fallback without ffmpeg: Pillow (per-frame adaptive palette, slower and larger)."""
    from PIL import Image

    ims = []
    for f in frames:
        im = Image.fromarray(f)
        if scale != 1.0:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        ims.append(im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG))
    ims[0].save(out_path, save_all=True, append_images=ims[1:], duration=round(1000 / fps), loop=0, optimize=False)


def export_episode(ep: Episode, view_keys: tuple[str, ...], fps: float, stride: int, scale: float,
                   out_path: Path, use_ffmpeg: bool) -> dict:
    views = [decode_range_pyav(*ep.videos[k], fps) for k in view_keys]
    n = min(v.shape[0] for v in views)
    if any(v.shape[0] != n for v in views):
        print(f"[ep{ep.index:03d}] WARNING: view lengths differ {[v.shape[0] for v in views]}; truncating to {n}")
    frames = np.concatenate([v[:n] for v in views], axis=2) if len(views) > 1 else views[0][:n]
    if n != ep.length:
        print(f"[ep{ep.index:03d}] WARNING: decoded {n} frames but metadata length is {ep.length}")
    frames = frames[::stride]
    out_fps = fps / stride
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.gif")
    (write_gif_ffmpeg if use_ffmpeg else write_gif_pillow)(frames, tmp, out_fps, scale)
    tmp.replace(out_path)
    return dict(episode_index=ep.index, task=ep.task, episode_length=ep.length, gif_frames=int(frames.shape[0]),
                gif_fps=out_fps, gif_size_hw=[int(frames.shape[1] * scale), int(frames.shape[2] * scale)],
                gif=str(out_path), bytes=out_path.stat().st_size)


def safe_dirname(text: str) -> str:
    return text.replace("/", "-").strip() or "untitled"


def main() -> int:
    storage = Path(os.environ.get("COSMOS_STORAGE", str(Path.home() / "project" / "cosmos_storage")))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=SUITES, default="libero_10", help="LIBERO suite (sets the defaults below)")
    p.add_argument("--libero-root", default=None, help="LeRobot v3 dir (default: $COSMOS_STORAGE/data/LIBERO_LeRobot_v3/<suite>)")
    p.add_argument("--episode-subset", default=None, help=f"subset JSON, path or bundled name (default: {SUBSET_JSON_TEMPLATE})")
    p.add_argument("--out-dir", default=None, help=f"where the per-task folders go (default: project/utils/{OUT_DIR_TEMPLATE})")
    p.add_argument("--views", choices=sorted(VIEWS), default="both", help="camera view(s); both = side by side")
    p.add_argument("--stride", type=int, default=1, help="keep every n-th frame (GIF fps = 20/n)")
    p.add_argument("--scale", type=float, default=1.0, help="spatial scale factor (1.0 = native 256 px per view)")
    p.add_argument("--tasks", default="all", help="comma-separated task indices of the subset JSON, or 'all'")
    p.add_argument("--only-episodes", default="all",
                   help="comma-separated episode indices to export (others in the subset are skipped), or 'all'")
    p.add_argument("--episodes", default=None,
                   help="comma-separated dataset episode indices to export INSTEAD of the subset (any episode of the suite; "
                        "the task folder comes from the dataset's task text)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--overwrite", action="store_true", help="re-encode GIFs that already exist")
    p.add_argument("--no-ffmpeg", action="store_true", help="force the Pillow GIF writer")
    p.add_argument("--force", action="store_true", help="skip the subset-JSON <-> dataset consistency check")
    args = p.parse_args()

    root = Path(args.libero_root) if args.libero_root else storage / "data" / "LIBERO_LeRobot_v3" / args.suite
    if not (root / "meta" / "info.json").is_file():
        print(f"ERROR: {root} is not a LeRobot dataset dir (meta/info.json missing). Set COSMOS_STORAGE / --libero-root.",
              file=sys.stderr)
        return 1
    subset_name = args.episode_subset or SUBSET_JSON_TEMPLATE.format(suite=args.suite)
    subset_path = find_episode_subset_json(subset_name)
    if subset_path is None:
        print(f"ERROR: episode subset {subset_name!r} not found", file=sys.stderr)
        return 1
    js = json.loads(subset_path.read_text())
    src = js.get("source_root_basename")
    if src and src != root.name and not args.force:
        print(f"ERROR: subset JSON was built from {src!r} but the dataset root is {root} (use --force to override)",
              file=sys.stderr)
        return 1
    wanted_tasks = None if args.tasks == "all" else {int(x) for x in args.tasks.split(",")}
    only_eps = None if args.only_episodes == "all" else {int(x) for x in args.only_episodes.split(",")}
    subset = {int(ti): v for ti, v in js["tasks"].items() if wanted_tasks is None or int(ti) in wanted_tasks}
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "utils" / OUT_DIR_TEMPLATE.format(suite=args.suite)

    use_ffmpeg = not args.no_ffmpeg and shutil.which("ffmpeg") is not None
    if not use_ffmpeg:
        print("[gif] ffmpeg not found (or --no-ffmpeg): using the Pillow writer")
    fps, episodes = load_libero_meta(root)
    by_index = {e.index: e for e in episodes}
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] root={root}\n[data] subset={subset_path}\n[out]  {out_dir}  views={args.views} stride={args.stride} scale={args.scale}")

    jobs = []
    task_index_of = {v["task"]: ti for ti, v in subset.items()}
    if args.episodes is not None:  # arbitrary episodes of the suite, independent of the subset
        for e in sorted({int(x) for x in args.episodes.split(",")}):
            if e not in by_index:
                print(f"ERROR: episode {e} is not in the dataset {root}", file=sys.stderr)
                return 1
            ep = by_index[e]
            out_path = out_dir / safe_dirname(ep.task) / f"ep{ep.index:03d}.gif"
            if out_path.is_file() and not args.overwrite:
                print(f"[skip] {out_path} exists")
                continue
            jobs.append((task_index_of.get(ep.task, -1), ep, out_path))
    for ti in sorted(subset) if args.episodes is None else []:
        task_text = subset[ti]["task"]
        for e in subset[ti]["episodes"]:
            if only_eps is not None and int(e) not in only_eps:
                continue
            ep = by_index[int(e)]
            if ep.task != task_text and not args.force:
                print(f"ERROR: episode {e} task text in the dataset {ep.task!r} != subset task {task_text!r}; "
                      f"wrong --libero-root / --episode-subset pairing? (use --force to override)", file=sys.stderr)
                return 1
            out_path = out_dir / safe_dirname(ep.task) / f"ep{ep.index:03d}.gif"
            if out_path.is_file() and not args.overwrite:
                print(f"[skip] {out_path} exists")
                continue
            jobs.append((ti, ep, out_path))

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(export_episode, ep, VIEWS[args.views], fps, args.stride, args.scale, out_path, use_ffmpeg): (ti, ep)
                for ti, ep, out_path in jobs}
        for fut in as_completed(futs):
            ti, ep = futs[fut]
            r = fut.result()
            r["task_index"] = ti
            results.append(r)
            print(f"[done] task{ti:02d} ep{ep.index:03d}  {r['gif_frames']} frames  {r['bytes'] / 1e6:.1f} MB  -> {r['gif']}")

    # manifest: merge with what is already on disk so repeated/partial runs stay complete
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest.update(dict(suite=root.name, libero_root=str(root), episode_subset=str(subset_path), views=args.views,
                         stride=args.stride, scale=args.scale, fps=fps,
                         only_episodes=sorted(only_eps) if only_eps is not None else "all"))
    if args.episodes is not None:
        extra = sorted(set(manifest.get("extra_episodes", [])) | {r["episode_index"] for r in results})
        manifest["extra_episodes"] = extra  # exported via --episodes, i.e. NOT (necessarily) members of the subset
    eps = {str(r["episode_index"]): r for r in manifest.get("episodes", {}).values()} if isinstance(manifest.get("episodes"), dict) else {}
    for r in results:
        eps[str(r["episode_index"])] = r
    manifest["episodes"] = dict(sorted(eps.items(), key=lambda kv: int(kv[0])))
    manifest["tasks"] = {str(ti): dict(task=subset[ti]["task"], episodes=subset[ti]["episodes"]) for ti in sorted(subset)}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    total = sum(r["bytes"] for r in results)
    print(f"[done] {len(results)} GIFs written ({total / 1e6:.1f} MB); manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
