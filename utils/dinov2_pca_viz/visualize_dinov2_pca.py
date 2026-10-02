#!/usr/bin/env python
"""LIBERO-10 x DINOv2 ViT-B/14: PCA visualization of the patch features and of the pooled REPA target.

One PNG per LIBERO-10 task, each describing ONE training window (the same 17-frame window the Cosmos3 LIBERO
dataset builds: consecutive frames at the native 20 fps, frame 0 = clean conditioning frame, frames 1..16 = the
frames the MoT denoises and the frozen DINOv2 teacher encodes). Three rows:

  row 1  the 16 raw frames the teacher sees (concat_view canvas: third-person | wrist, 256 px each)
  row 2  DINOv2 patch tokens of every frame -> PCA (3 comps) -> RGB, 16x16 patches per view (224 px input)
  row 3  the distillation target of cosmos_hs10/hs11 ("avgpool" adapter = F.adaptive_avg_pool3d over
         (T, H, W) -> target_grid, default 16x16x16 -> 4x5x5 per view, i.e. the 4x5x10 MoT token grid), projected
         with the SAME PCA basis / color scaling, one map per temporal bin (4 frames -> 1 by default)

Mirrors cosmos_hs11 `model/generator/repa/dinov2_teacher.py` (preprocessing, post-LayerNorm patch tokens) and
`adapters.py::adaptive_pool_teacher_grid` (pooling), but is standalone: only torch, transformers, av, pandas,
pyarrow, numpy, matplotlib are needed (all in the `cosmos3-pt` conda env). Video decoding uses PyAV (libdav1d), not
torchcodec, so it also runs on the login node.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent  # <...>/project

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_KEY = "observation.images.image"  # third-person camera (left half of the concat_view canvas)
WRIST_KEY = "observation.images.wrist_image"  # wrist camera (right half)
VIEW_KEYS = (IMAGE_KEY, WRIST_KEY)
VIEW_NAMES = ("3rd-person", "wrist")

TEACHERS = {
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
}
DEFAULT_SUBSET_JSON = "libero_10_3ep_per_task_seed42.json"  # the 30-demo few-shot training subset

# figure colors (text wears text tokens, the data wears its own RGB)
C_TEXT = "#1f2937"
C_MUTED = "#6b7280"
C_SPINE = "#d1d5db"
C_SEP = "#ffffff"


# ----------------------------------------------------------------------------------------------------------------
# LIBERO-10 LeRobot v3 metadata + PyAV decoding
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Episode:
    index: int
    task_index: int
    task: str
    length: int
    videos: dict[str, tuple[str, float]] = field(default_factory=dict)  # view key -> (mp4 path, from_timestamp)


def load_libero_meta(root: Path) -> tuple[float, dict[int, str], list[Episode]]:
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    tasks_df = pd.read_parquet(root / "meta" / "tasks.parquet")
    tasks = {int(v): str(k) for k, v in tasks_df["task_index"].items()}  # index = task text in LeRobot v3
    text_to_index = {v: k for k, v in tasks.items()}
    files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {root / 'meta' / 'episodes'}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    episodes: list[Episode] = []
    for _, row in df.iterrows():
        task_text = str(list(row["tasks"])[0])
        ep = Episode(
            index=int(row["episode_index"]),
            task_index=text_to_index[task_text],
            task=task_text,
            length=int(row["length"]),
        )
        for key in VIEW_KEYS:
            rel = info["video_path"].format(
                video_key=key,
                chunk_index=int(row[f"videos/{key}/chunk_index"]),
                file_index=int(row[f"videos/{key}/file_index"]),
            )
            ep.videos[key] = (str(root / rel), float(row[f"videos/{key}/from_timestamp"]))
        episodes.append(ep)
    episodes.sort(key=lambda e: e.index)
    return fps, tasks, episodes


def find_episode_subset_json(name_or_path: str) -> Path | None:
    """Resolve the few-shot subset JSON: an explicit path, or the bundled file of the newest cosmos_hs* repo."""
    p = Path(name_or_path).expanduser()
    if p.is_file():
        return p
    pattern = str(PROJECT_ROOT / "cosmos_hs*" / "packages" / "cosmos3" / "cosmos_framework" / "data" / "generator"
                  / "action" / "episode_subsets" / name_or_path)
    hits = sorted(glob.glob(pattern), reverse=True)
    return Path(hits[0]) if hits else None


def decode_frames_pyav(path: str, timestamps: list[float], tol: float) -> np.ndarray:
    """Decode the frames at the given absolute timestamps (seconds in the concatenated mp4) -> [T,H,W,3] uint8."""
    import av

    out: list[np.ndarray] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = stream.time_base
        # land on a keyframe comfortably before the first target, then walk forward
        container.seek(max(0, int((timestamps[0] - 1.0) / tb)), stream=stream, backward=True, any_frame=False)
        idx = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            ts = float(frame.pts * tb)
            if ts < timestamps[idx] - tol:
                continue
            if ts > timestamps[idx] + tol:
                raise RuntimeError(
                    f"{path}: no frame within {tol:.3f}s of t={timestamps[idx]:.3f}s (decoder is at {ts:.3f}s)"
                )
            out.append(frame.to_ndarray(format="rgb24"))
            idx += 1
            if idx == len(timestamps):
                break
    if len(out) != len(timestamps):
        raise RuntimeError(f"{path}: decoded {len(out)}/{len(timestamps)} frames (end of file?)")
    return np.stack(out, axis=0)


# ----------------------------------------------------------------------------------------------------------------
# DINOv2 teacher (same recipe as cosmos_hs11 dinov2_teacher.py) + avgpool target adapter
# ----------------------------------------------------------------------------------------------------------------
def load_dinov2(hf_repo: str, device: torch.device):
    from transformers import Dinov2Model

    try:
        model = Dinov2Model.from_pretrained(hf_repo, local_files_only=True)
        print(f"[model] {hf_repo}: loaded from the local HF cache ({os.environ.get('HF_HUB_CACHE', '$HF_HOME/hub')})")
    except Exception as err:  # noqa: BLE001 - fall back to the hub (needs internet)
        print(f"[model] {hf_repo} not in the local HF cache ({type(err).__name__}); downloading ...")
        model = Dinov2Model.from_pretrained(hf_repo)
    model.requires_grad_(False).eval().to(device)
    return model


def preprocess_frames(frames_uint8: torch.Tensor, size: int) -> torch.Tensor:
    """[N,3,H,W] uint8 -> normalized float32 at size x size (bilinear + antialias, ImageNet mean/std)."""
    x = frames_uint8.float() / 255.0
    if tuple(x.shape[-2:]) != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def encode_frames(model, frames_uint8: torch.Tensor, size: int, device: torch.device, dtype: str, batch: int) -> torch.Tensor:
    """[N,3,H,W] uint8 -> [N,hp,wp,D] float32 (CPU) post-LayerNorm patch tokens (CLS / registers dropped)."""
    n_prefix = 1 + int(getattr(model.config, "num_register_tokens", 0) or 0)
    side = size // int(model.config.patch_size)
    outs = []
    for start in range(0, frames_uint8.shape[0], batch):
        x = preprocess_frames(frames_uint8[start : start + batch].to(device), size)
        if device.type == "cuda" and dtype in ("bf16", "fp16"):
            with torch.autocast("cuda", dtype=torch.bfloat16 if dtype == "bf16" else torch.float16):
                hidden = model(pixel_values=x).last_hidden_state
        else:
            hidden = model(pixel_values=x).last_hidden_state
        tokens = hidden[:, n_prefix:]
        if tokens.shape[1] != side * side:
            raise RuntimeError(f"DINOv2 returned {tokens.shape[1]} patch tokens, expected {side}x{side}")
        outs.append(tokens.reshape(tokens.shape[0], side, side, -1).float().cpu())
    return torch.cat(outs, dim=0)


def avgpool_target(tokens: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
    """[T,hp,wp,D] -> [T',h',w',D]: the parameter-free "avgpool" REPA target adapter (adaptive_avg_pool3d)."""
    x = tokens.permute(3, 0, 1, 2).unsqueeze(0).float()  # [1,D,T,H,W]
    return F.adaptive_avg_pool3d(x, grid)[0].permute(1, 2, 3, 0).contiguous()


def adaptive_bins(n_in: int, n_out: int) -> list[tuple[int, int]]:
    """The [start, end) input bins of adaptive average pooling n_in -> n_out (e.g. 16->4: 4 consecutive each)."""
    return [(math.floor(i * n_in / n_out), math.ceil((i + 1) * n_in / n_out)) for i in range(n_out)]


# ----------------------------------------------------------------------------------------------------------------
# PCA -> RGB
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class PCA:
    mean: torch.Tensor  # [D]
    components: torch.Tensor  # [D,3]
    lo: torch.Tensor  # [3] low percentile of the fitted projections (-> 0)
    hi: torch.Tensor  # [3] high percentile (-> 1)
    explained: list[float]  # variance ratio of the 3 components

    def project(self, tokens: torch.Tensor) -> torch.Tensor:
        """[...,D] -> [...,3] in [0,1] (same basis + same color scaling for everything that is passed in)."""
        z = (tokens.float() - self.mean) @ self.components
        return ((z - self.lo) / (self.hi - self.lo).clamp_min(1e-8)).clamp_(0.0, 1.0)


def fit_pca(tokens: torch.Tensor, percentiles: tuple[float, float] = (1.0, 99.0)) -> PCA:
    x = tokens.reshape(-1, tokens.shape[-1]).float()
    mean = x.mean(dim=0)
    xc = x - mean
    _, s, vh = torch.linalg.svd(xc, full_matrices=False)
    comps = vh[:3].T.contiguous()  # [D,3]
    # fix the sign so that each component is positively correlated with the token with the largest norm change
    # (purely cosmetic; any fixed sign convention works)
    sign = torch.sign((xc @ comps).sum(dim=0))
    sign[sign == 0] = 1.0
    comps = comps * sign
    z = xc @ comps
    lo = torch.quantile(z, percentiles[0] / 100.0, dim=0)
    hi = torch.quantile(z, percentiles[1] / 100.0, dim=0)
    var = s**2
    explained = (var[:3] / var.sum()).tolist()
    return PCA(mean=mean, components=comps, lo=lo, hi=hi, explained=explained)


# ----------------------------------------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------------------------------------
def _style_image_axes(ax, view_boundary_x: float | None):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if view_boundary_x is not None:
        ax.axvline(view_boundary_x, color=C_SEP, lw=1.0, alpha=0.9)


def render_task_figure(
    out_path: Path,
    frames: np.ndarray,  # [T,H,2W,3] uint8
    feat_rgb: np.ndarray,  # [T,hp,2wp,3] in [0,1]
    pooled_rgb: np.ndarray,  # [T',h',2w',3] in [0,1]
    bins: list[tuple[int, int]],
    frame_ids: list[int],
    title: str,
    subtitle: str,
    row_labels: list[str],
    dpi: int,
    cell_w_in: float = 2.4,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    T, H, W2, _ = frames.shape
    W = W2 // 2
    Tp = pooled_rgb.shape[0]
    cell_h_in = cell_w_in * H / W2
    left_in, right_in, top_in, bottom_in, gap_in = 2.9, 0.3, 1.05, 0.55, 0.32
    fig_w = left_in + T * cell_w_in + right_in
    fig_h = top_in + 3 * cell_h_in + 2 * gap_in + bottom_in
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    outer = GridSpec(
        3, 1, figure=fig,
        left=left_in / fig_w, right=1 - right_in / fig_w, top=1 - top_in / fig_h, bottom=bottom_in / fig_h,
        hspace=gap_in / cell_h_in,
    )
    grids = [
        GridSpecFromSubplotSpec(1, T, subplot_spec=outer[0], wspace=0.06),
        GridSpecFromSubplotSpec(1, T, subplot_spec=outer[1], wspace=0.06),
        GridSpecFromSubplotSpec(1, Tp, subplot_spec=outer[2], wspace=0.06),
    ]

    # row 1: raw frames
    for c in range(T):
        ax = fig.add_subplot(grids[0][0, c])
        ax.imshow(frames[c], extent=(0, W2, H, 0), interpolation="bilinear")
        _style_image_axes(ax, W)
        ax.set_title(f"frame {frame_ids[c]}", fontsize=9, color=C_MUTED, pad=3)
    # row 2: per-frame PCA maps
    for c in range(T):
        ax = fig.add_subplot(grids[1][0, c])
        ax.imshow(feat_rgb[c], extent=(0, W2, H, 0), interpolation="nearest")
        _style_image_axes(ax, W)
    # row 3: pooled maps, one per temporal bin, drawn at frame size and centered under the frames of the bin
    for i in range(Tp):
        b0, b1 = bins[i]
        span = max(1, b1 - b0)
        ax = fig.add_subplot(grids[2][0, i])
        ax.set_xlim(0, span * W2)
        ax.set_ylim(H, 0)
        x0 = (span - 1) * W
        ax.imshow(pooled_rgb[i], extent=(x0, x0 + W2, H, 0), interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(C_SPINE)
            sp.set_linewidth(0.8)
            sp.set_linestyle((0, (4, 3)))
        ax.axvline(x0 + W, color=C_SEP, lw=1.0, alpha=0.9)
        ax.set_xlabel(f"frames {frame_ids[b0]}-{frame_ids[b1 - 1]}  ->  latent t={i + 1}", fontsize=9, color=C_MUTED, labelpad=4)

    # row labels (left margin), centered on each outer row
    for r, label in enumerate(row_labels):
        bbox = outer[r].get_position(fig)
        fig.text(0.012, (bbox.y0 + bbox.y1) / 2, label, ha="left", va="center", fontsize=10, color=C_TEXT, linespacing=1.4)
    fig.text(left_in / fig_w, 1 - 0.28 / fig_h, title, ha="left", va="top", fontsize=13, color=C_TEXT, weight="bold")
    fig.text(left_in / fig_w, 1 - 0.62 / fig_h, subtitle, ha="left", va="top", fontsize=9.5, color=C_MUTED)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------
def slugify(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:max_len].rstrip("_")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = os.environ.get("LIBERO_ROOT") or str(
        Path(os.environ.get("COSMOS_STORAGE", str(PROJECT_ROOT / "cosmos_storage"))) / "data" / "LIBERO_LeRobot_v3" / "libero_10"
    )
    p.add_argument("--libero-root", default=default_root, help="local LeRobot v3 dir of LIBERO-10 (meta/, data/, videos/)")
    p.add_argument("--out-dir", default=str(HERE / "results"), help="results root; PNGs go to <out-dir>/<run-name>/")
    p.add_argument("--run-name", default=None, help="sub-folder name (default: <teacher>_in<size>_pool<T>x<H>x<W>[_refit])")
    p.add_argument("--teacher", default="dinov2_vitb14", choices=sorted(TEACHERS), help="DINOv2 variant (HF transformers)")
    p.add_argument("--input-size", type=int, default=224, help="teacher input size per view (224 -> 16x16 patches of 14px)")
    p.add_argument("--num-frames", type=int, default=16, help="frames the teacher encodes (window = num_frames + 1 cond frame)")
    p.add_argument("--target-grid", type=int, nargs=3, default=(4, 5, 5), metavar=("T", "H", "W"),
                   help="per-view avgpool target grid of the REPA loss (cosmos_hs10/hs11 default 4 5 5)")
    p.add_argument("--tasks", default="all", help="comma-separated LIBERO-10 task indices (default: all 10)")
    p.add_argument("--episode-subset", default=DEFAULT_SUBSET_JSON,
                   help="few-shot subset JSON (path or bundled name) whose episodes are preferred; 'none' = any episode of the task")
    p.add_argument("--episode-pick", default="first", choices=("first", "random"),
                   help="which episode of the task (within the subset if given): the lowest index, or a seeded random one")
    p.add_argument("--window-pos", default="0.5",
                   help="where the window starts inside the episode: fraction in [0,1] of the valid starts, or 'random'")
    p.add_argument("--start-frame", type=int, default=None, help="explicit window start frame for every task (overrides --window-pos)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pca-scope", default="window", choices=("window", "global"),
                   help="fit the PCA per task window (default) or once on the tokens of all tasks (comparable colors across PNGs)")
    p.add_argument("--pca-per-view", action="store_true",
                   help="fit a separate PCA per camera view (3rd-person / wrist) instead of one joint PCA over both views")
    p.add_argument("--pooled-pca", default="shared", choices=("shared", "refit"),
                   help="project the pooled target with the window's PCA basis+scaling (default) or refit a PCA on the pooled tokens")
    p.add_argument("--percentiles", type=float, nargs=2, default=(1.0, 99.0), help="robust color scaling of each PC")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda[:i]")
    p.add_argument("--teacher-dtype", default="bf16", choices=("bf16", "fp16", "fp32"), help="autocast dtype on CUDA (CPU is fp32)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: OMP_NUM_THREADS / torch default)")
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--save-npz", action="store_true", help="also save the raw tokens / pooled targets (float16) per task")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t_start = time.time()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    root = Path(args.libero_root).expanduser()
    if not (root / "meta" / "info.json").is_file():
        print(f"ERROR: {root} is not a LeRobot dataset dir (meta/info.json missing). Set LIBERO_ROOT / --libero-root.", file=sys.stderr)
        return 1
    grid = tuple(int(v) for v in args.target_grid)
    run_name = args.run_name or f"{args.teacher}_in{args.input_size}_pool{grid[0]}x{grid[1]}x{grid[2]}" + (
        "_refit" if args.pooled_pca == "refit" else ""
    ) + ("_globalpca" if args.pca_scope == "global" else "") + ("_perview" if args.pca_per_view else "")
    out_dir = Path(args.out_dir).expanduser() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    fps, tasks, episodes = load_libero_meta(root)
    print(f"[data] {root}: {len(episodes)} episodes, {len(tasks)} tasks, {fps:g} fps, device={device}")
    task_ids = sorted(tasks) if args.tasks == "all" else [int(v) for v in args.tasks.split(",")]

    subset: dict[int, list[int]] = {}
    subset_path = None
    if args.episode_subset.lower() != "none":
        subset_path = find_episode_subset_json(args.episode_subset)
        if subset_path is None:
            print(f"[data] WARNING: episode subset {args.episode_subset!r} not found; using any episode of each task")
        else:
            js = json.loads(subset_path.read_text())
            subset = {int(k): [int(e) for e in v["episodes"]] for k, v in js["tasks"].items()}
            print(f"[data] episode subset: {subset_path}")

    window = args.num_frames + 1  # + the conditioning frame 0
    tol = 0.5 / fps
    model = load_dinov2(TEACHERS[args.teacher], device)
    patch = int(model.config.patch_size)
    if args.input_size % patch:
        print(f"ERROR: --input-size must be a multiple of the patch size {patch}", file=sys.stderr)
        return 1
    side = args.input_size // patch
    bins_t = adaptive_bins(args.num_frames, grid[0])

    # ---- pass 1: pick windows, decode, encode, pool ------------------------------------------------------------
    records: list[dict] = []
    for ti in task_ids:
        candidates = [e for e in episodes if e.task_index == ti and e.length >= window]
        if not candidates:
            print(f"[task {ti}] no episode with >= {window} frames; skipping")
            continue
        if ti in subset:
            in_subset = [e for e in candidates if e.index in subset[ti]]
            if in_subset:
                candidates = sorted(in_subset, key=lambda e: subset[ti].index(e.index))
        ep = rng.choice(candidates) if args.episode_pick == "random" else candidates[0]
        n_valid = ep.length - window  # valid window starts: 0 .. n_valid (same as the training dataset)
        if args.start_frame is not None:
            start = min(max(0, args.start_frame), n_valid)
        elif args.window_pos == "random":
            start = rng.randint(0, n_valid)
        else:
            start = int(round(float(args.window_pos) * n_valid))
        rel_ts = [(start + i) / fps for i in range(window)]
        t0 = time.time()
        views = []
        for key in VIEW_KEYS:
            path, from_ts = ep.videos[key]
            views.append(decode_frames_pyav(path, [from_ts + t for t in rel_ts], tol))  # [window,H,W,3]
        clip = np.stack(views, axis=1)[1:]  # drop the conditioning frame -> [T, V, H, W, 3] (teacher frames only)
        t_dec = time.time() - t0
        T, V, H, W, _ = clip.shape
        frames_t = torch.from_numpy(clip).permute(0, 1, 4, 2, 3).reshape(T * V, 3, H, W)  # [T*V,3,H,W] uint8
        t0 = time.time()
        tokens = encode_frames(model, frames_t, args.input_size, device, args.teacher_dtype, args.batch_size)
        tokens = tokens.reshape(T, V, side, side, -1).permute(1, 0, 2, 3, 4).contiguous()  # [V,T,hp,wp,D]
        pooled = torch.stack([avgpool_target(tokens[v], grid) for v in range(V)], dim=0)  # [V,T',h',w',D]
        t_enc = time.time() - t0
        print(
            f"[task {ti}] ep {ep.index} (len {ep.length}) start {start}: frames {start + 1}-{start + T} | "
            f"decode {t_dec:.1f}s, encode+pool {t_enc:.1f}s | tokens {tuple(tokens.shape)} -> pooled {tuple(pooled.shape)}"
        )
        records.append(dict(task_index=ti, episode=ep, start=start, clip=clip, tokens=tokens, pooled=pooled))

    if not records:
        print("ERROR: nothing to visualize", file=sys.stderr)
        return 1

    # ---- pass 2: PCA + figures -------------------------------------------------------------------------------
    pct = (float(args.percentiles[0]), float(args.percentiles[1]))
    n_views = records[0]["tokens"].shape[0]

    def fit_pcas(token_sets: list[torch.Tensor]) -> list[PCA]:
        """One PCA per view (--pca-per-view) or one joint PCA repeated for every view. token_sets: [V,...,D] tensors."""
        if args.pca_per_view:
            return [fit_pca(torch.cat([t[v].reshape(-1, t.shape[-1]) for t in token_sets]), pct) for v in range(n_views)]
        joint = fit_pca(torch.cat([t.reshape(-1, t.shape[-1]) for t in token_sets]), pct)
        return [joint] * n_views

    global_pcas = fit_pcas([r["tokens"] for r in records]) if args.pca_scope == "global" else None
    run_info = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        args={k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
        run_name=run_name,
        teacher_hf_repo=TEACHERS[args.teacher],
        patch_grid_per_view=[args.num_frames, side, side],
        target_grid_per_view=list(grid),
        temporal_bins=[[b0, b1] for b0, b1 in bins_t],
        episode_subset=str(subset_path) if subset_path else None,
        fps=fps,
        device=str(device),
        tasks=[],
    )
    for r in records:
        ti, ep, start = r["task_index"], r["episode"], r["start"]
        tokens, pooled, clip = r["tokens"], r["pooled"], r["clip"]
        V, T = tokens.shape[:2]
        pcas = global_pcas if global_pcas is not None else fit_pcas([tokens])
        feat_rgb = torch.stack([pcas[v].project(tokens[v]) for v in range(V)])  # [V,T,hp,wp,3]
        pooled_pcas = fit_pcas([pooled]) if args.pooled_pca == "refit" else pcas
        pooled_rgb = torch.stack([pooled_pcas[v].project(pooled[v]) for v in range(V)])  # [V,T',h',w',3]
        pca, pooled_pca = pcas[0], pooled_pcas[0]  # for the subtitle; per-view numbers below

        def ev_text(ps: list[PCA]) -> str:
            if args.pca_per_view:
                return " | ".join(f"{VIEW_NAMES[v]} {ps[v].explained[0]:.0%}/{ps[v].explained[1]:.0%}/{ps[v].explained[2]:.0%}" for v in range(V))
            return f"{ps[0].explained[0]:.0%}/{ps[0].explained[1]:.0%}/{ps[0].explained[2]:.0%}"
        # concat_view layout: views side by side along width (third-person left, wrist right)
        frames_cat = np.concatenate([clip[:, v] for v in range(V)], axis=2)  # [T,H,V*W,3]
        feat_cat = torch.cat([feat_rgb[v] for v in range(V)], dim=2).numpy()  # [T,hp,V*wp,3]
        pooled_cat = torch.cat([pooled_rgb[v] for v in range(V)], dim=2).numpy()  # [T',h',V*w',3]
        frame_ids = [start + 1 + i for i in range(T)]
        task_text = tasks[ti]
        title = f"LIBERO-10 task {ti:02d}:  \"{task_text}\""
        n_fit = (len(records) if global_pcas is not None else 1) * (1 if args.pca_per_view else V) * T * side * side
        ev = pca.explained
        pca_kind = f"{args.pca_scope}, {'per view' if args.pca_per_view else 'both views jointly'}"
        pooled_note = (
            "same PCA basis + color scaling as row 2" if args.pooled_pca == "shared"
            else f"own PCA (PC1-3 explain {ev_text(pooled_pcas)})"
        )
        subtitle = (
            f"episode {ep.index} (length {ep.length}, {fps:g} fps) · window start {start}: frame {start} = clean conditioning frame (not encoded), "
            f"frames {start + 1}-{start + T} = predicted frames shown here · {args.teacher} @ {args.input_size}px -> {side}x{side} patches per view "
            f"· PCA fit on {n_fit} tokens ({pca_kind}), PC1/2/3 explain {ev_text(pcas)} of the variance "
            f"· colors: PC1->R, PC2->G, PC3->B, {pct[0]:g}-{pct[1]:g} percentile stretch · row 3: {pooled_note}"
        )
        row_labels = [
            f"input frames\n{VIEW_NAMES[0]} | {VIEW_NAMES[1]}\n({H}x{W} px each)",
            f"DINOv2 patch tokens\nPCA -> RGB\n({side}x{side} per view)",
            f"REPA avgpool target\n{grid[0]}x{grid[1]}x{grid[2]} per view\n(1 map / {args.num_frames // grid[0] if args.num_frames % grid[0] == 0 else 'adaptive'} frames)",
        ]
        png = out_dir / f"task{ti:02d}_{slugify(task_text)}.png"
        render_task_figure(png, frames_cat, feat_cat, pooled_cat, bins_t, frame_ids, title, subtitle, row_labels, args.dpi)
        entry = dict(
            task_index=ti, task=task_text, episode_index=ep.index, episode_length=ep.length, window_start=start,
            conditioning_frame=start, teacher_frames=[frame_ids[0], frame_ids[-1]],
            pca_explained_variance=[[round(x, 4) for x in pcas[v].explained] for v in range(V)] if args.pca_per_view
            else [round(x, 4) for x in ev],
            png=str(png),
        )
        if args.save_npz:
            npz = out_dir / f"task{ti:02d}_{slugify(task_text)}_features.npz"
            np.savez_compressed(
                npz, tokens=tokens.half().numpy(), pooled=pooled.half().numpy(), frames=clip,
                pca_components=np.stack([q.components.numpy() for q in pcas]), pca_mean=np.stack([q.mean.numpy() for q in pcas]),
                frame_ids=np.array(frame_ids),
                views=np.array(VIEW_NAMES),
            )
            entry["npz"] = str(npz)
        run_info["tasks"].append(entry)
        print(f"[task {ti}] -> {png}")

    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"[done] {len(run_info['tasks'])} PNGs in {out_dir} ({time.time() - t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
