# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Download RoboCasa365 datasets / kitchen assets from the official Box links (or verified HF mirrors) -- stdlib only.

A stand-alone, resumable replacement for ``robocasa.scripts.download_datasets`` and
``robocasa.scripts.download_kitchen_assets`` that does NOT import the ``robocasa`` package (so it runs in any
Python >= 3.12 on a node with internet, e.g. the login node, before the simulator env exists). It reads the same
link tables the official scripts use (``robocasa/models/assets/box_links/box_links_{ds,assets}.json`` of a robocasa
clone) and writes the same on-disk layout:

  datasets:  <dest>/<split>/<kind>/<Task>/<date>/lerobot/        (LeRobot v2.1 export + extras/dataset_meta.json)
             = robocasa's ``DATASET_BASE_PATH/v1.0/...`` when <dest> = ``$ROBOCASA_DATA/v1.0``
  assets:    <clone>/robocasa/models/assets/{textures,generative_textures,fixtures,objects/{objaverse,aigen_objs,lightwheel}}

Examples (see examples/setup_robocasa365.sh):

  python -m cosmos_framework.scripts.download_robocasa365 datasets \\
      --box-links ~/project/robocasa/robocasa/models/assets/box_links/box_links_ds.json \\
      --dest $COSMOS_STORAGE/data/robocasa365/v1.0 --split target --kind atomic --tasks CloseFridge OpenDrawer ...

  python -m cosmos_framework.scripts.download_robocasa365 assets \\
      --box-links ~/project/robocasa/robocasa/models/assets/box_links/box_links_assets.json \\
      --assets-dir ~/project/robocasa/robocasa/models/assets --types all

Every archive is fetched to ``<name>.part`` first (retried), extracted with Python's ``data`` filter (no absolute
paths / links escaping the destination), then deleted. A dataset whose ``lerobot/meta/info.json`` already exists is
skipped, so re-running after an interruption resumes.

``--source`` (default ``auto``): some clusters reset every TCP connection to utexas.box.com ("[Errno 104] Connection
reset by peer" on all archives, seen 2026-11-02). ``auto`` tries Box first and falls back to the Hugging Face mirrors
below for the rest of the run; ``hf`` skips Box; ``box`` never falls back. The mirrors were checked against Box on
2026-11-02: the 6 asset zips are byte-identical (size + first/last MiB sha256), the dataset tars differ only in tar
metadata (one task extracted and every file sha256-compared against the Box copy: identical).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

# Asset registry key (box_links_assets.json) -> folder under <assets-dir>, as in robocasa.scripts.download_kitchen_assets
# (the zip is extracted into the folder's PARENT, i.e. the archive carries the folder itself). ``--types`` uses the
# official short names.
ASSET_TYPES: dict[str, tuple[str, str]] = {
    "tex": ("textures", "textures"),
    "tex_generative": ("generative_textures", "generative_textures"),
    "fixtures_lw": ("fixtures_lightwheel", "fixtures"),
    "objs_objaverse": ("objaverse", "objects/objaverse"),
    "objs_aigen": ("aigen_objs", "objects/aigen_objs"),
    "objs_lw": ("objects_lightwheel", "objects/lightwheel"),
}


def direct_url(shared_url: str, ext: str) -> str:
    """``https://utexas.box.com/s/<id>`` -> ``https://utexas.box.com/shared/static/<id>.<ext>``."""
    shared_id = shared_url.rstrip("/").split("/")[-1]
    base = shared_url.split("/s/")[0]
    return f"{base}/shared/static/{shared_id}.{ext}"


# Hugging Face mirrors of the Box archives (public, ungated; see the module docstring for how they were verified).
HF_ASSETS_REPO = "twilighted/Robocasa365-Assets"  # one zip per Box asset key, byte-identical to Box
HF_ASSET_FILES: dict[str, str] = {  # box_links_assets.json key -> file in HF_ASSETS_REPO
    "textures": "textures.zip",
    "generative_textures": "generative_textures.zip",
    "fixtures_lightwheel": "fixtures.zip",
    "objaverse": "objaverse.zip",
    "aigen_objs": "aigen_objs.zip",
    "objects_lightwheel": "lightwheel.zip",
}
HF_DATASETS_REPO = "huiwon/robocasa365"  # mirrors the Box tree: <split>/<kind>/<Task>/<date>/lerobot.tar
SOURCES = ("auto", "box", "hf")


def hf_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def hf_asset_url(link_key: str, repo: str = HF_ASSETS_REPO) -> str:
    return hf_url(repo, HF_ASSET_FILES[link_key])


def hf_dataset_url(key: str, repo: str = HF_DATASETS_REPO) -> str:
    """``key`` is the box_links_ds.json key, e.g. ``target/atomic/CloseFridge/20250816/lerobot.tar``."""
    return hf_url(repo, key)


def source_order(source: str, box: str, hf: str) -> list[tuple[str, str]]:
    """Candidate (label, url) pairs for ``--source``; ``auto`` = Box then HF."""
    if source == "box":
        return [("box", box)]
    if source == "hf":
        return [("hf", hf)]
    return [("box", box), ("hf", hf)]


def _fmt(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def download(url: str, dst: Path, *, tries: int = 3, chunk: int = 1 << 22) -> None:
    """Stream ``url`` to ``dst`` via ``dst.with_suffix('.part')`` with a byte-level progress line."""
    part = dst.with_name(dst.name + ".part")
    last_err: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cosmos-framework/robocasa365-downloader"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(part, "wb") as fh:
                total = int(resp.headers.get("Content-Length") or 0)
                done, t0, t_last = 0, time.time(), 0.0
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
                    done += len(buf)
                    now = time.time()
                    if now - t_last > 5:
                        rate = done / max(now - t0, 1e-6)
                        pct = f" {100 * done / total:5.1f}%" if total else ""
                        print(f"    {dst.name}: {_fmt(done)}/{_fmt(total)}{pct} @ {_fmt(rate)}/s", flush=True)
                        t_last = now
            if total and done != total:
                raise OSError(f"short read: {done} of {total} bytes")
            os.replace(part, dst)
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"    attempt {attempt}/{tries} failed: {exc}", flush=True)
            time.sleep(5 * attempt)
    part.unlink(missing_ok=True)
    raise RuntimeError(f"download failed after {tries} attempts: {url}") from last_err


_PREFERRED: list[str] = []  # labels that succeeded after an earlier one failed, most recent first


def download_any(candidates: list[tuple[str, str]], dst: Path, *, tries: int = 3) -> str:
    """Try each (label, url) with :func:`download`; return the label that worked.

    A source that succeeds after another failed is moved to the front for the rest of the process, so a run on a
    network that resets every Box connection pays the Box timeouts once, not once per archive.
    """
    order = sorted(candidates, key=lambda c: _PREFERRED.index(c[0]) if c[0] in _PREFERRED else len(_PREFERRED))
    errors: list[str] = []
    for i, (label, url) in enumerate(order):
        if i > 0:
            print(f"    falling back to {label}: {url}", flush=True)
        try:
            download(url, dst, tries=tries)
        except RuntimeError as exc:
            errors.append(f"{label}: {exc}")
            continue
        if i > 0 and label not in _PREFERRED:
            _PREFERRED.insert(0, label)
        return label
    raise RuntimeError("all sources failed: " + " | ".join(errors))


def extract(archive: Path, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".tar":
        with tarfile.open(archive, "r") as tar:
            tar.extractall(path=into, filter="data")
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                target = (into / member.filename).resolve()
                if not str(target).startswith(str(into.resolve())):
                    raise RuntimeError(f"unsafe zip member {member.filename!r} in {archive}")
            zf.extractall(path=into)
    else:
        raise ValueError(f"unknown archive type: {archive}")


# ------------------------------------------------------------------------------------------------ datasets
def cmd_datasets(args: argparse.Namespace) -> int:
    links: dict[str, str] = json.loads(Path(args.box_links).read_text())
    prefix = f"{args.split}/{args.kind}/"
    # key = "<split>/<kind>/<Task>/<date>/lerobot.tar"; one dated export per task in the official table.
    by_task: dict[str, list[str]] = {}
    for key in links:
        if key.startswith(prefix) and key.endswith("/lerobot.tar"):
            by_task.setdefault(key.split("/")[2], []).append(key)
    tasks = args.tasks or sorted(by_task)
    missing = [t for t in tasks if t not in by_task]
    if missing:
        print(f"ERROR: no {prefix}<task>/<date>/lerobot.tar link for: {missing}", file=sys.stderr)
        return 1
    dest = Path(args.dest)
    failed: list[str] = []
    for i, task in enumerate(tasks, 1):
        for key in sorted(by_task[task]):
            rel = Path(key).with_suffix("")  # <split>/<kind>/<Task>/<date>/lerobot
            ds_dir = dest / rel
            date_dir = ds_dir.parent
            if (ds_dir / "meta" / "info.json").is_file():
                print(f"[{i}/{len(tasks)}] {task}: already present at {ds_dir}, skipping", flush=True)
                continue
            candidates = source_order(
                args.source, direct_url(links[key], "tar"), hf_dataset_url(key, args.hf_datasets_repo)
            )
            print(
                f"[{i}/{len(tasks)}] {task}: {key}\n    " + "\n    ".join(f"{lbl}: {u}" for lbl, u in candidates),
                flush=True,
            )
            if args.dryrun:
                continue
            date_dir.mkdir(parents=True, exist_ok=True)
            tar_path = date_dir / "lerobot.tar"
            try:
                download_any(candidates, tar_path)
                if ds_dir.exists():  # a previous partial extraction
                    shutil.rmtree(ds_dir)
                print(f"    extracting into {date_dir}", flush=True)
                extract(tar_path, date_dir)
                if not (ds_dir / "meta" / "info.json").is_file():
                    raise RuntimeError(
                        f"archive did not produce {ds_dir}/meta/info.json; contents: {os.listdir(date_dir)}"
                    )
                info = json.loads((ds_dir / "meta" / "info.json").read_text())
                print(
                    f"    OK {task}: {info.get('total_episodes')} episodes, {info.get('total_frames')} frames, "
                    f"{info.get('fps')} fps, codebase {info.get('codebase_version')}, extras: {(ds_dir / 'extras').is_dir()}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    FAILED {task}: {exc}", file=sys.stderr, flush=True)
                failed.append(task)
            finally:
                tar_path.unlink(missing_ok=True)
    if failed:
        print(f"ERROR: {len(failed)} task(s) failed: {failed} -- re-run to retry", file=sys.stderr)
        return 1
    print(f"done: {len(tasks)} task(s) under {dest / prefix}")
    return 0


# ------------------------------------------------------------------------------------------------ assets
def cmd_assets(args: argparse.Namespace) -> int:
    links: dict[str, str] = json.loads(Path(args.box_links).read_text())
    types = list(ASSET_TYPES) if "all" in args.types else args.types
    assets_dir = Path(args.assets_dir)
    failed: list[str] = []
    for i, t in enumerate(types, 1):
        link_key, folder = ASSET_TYPES[t]
        target = assets_dir / folder
        marker = target / ".cosmos_download_complete"
        if marker.is_file():
            print(f"[{i}/{len(types)}] {t}: already present at {target}, skipping", flush=True)
            continue
        url = direct_url(links[link_key], "zip")
        candidates = source_order(args.source, url, hf_asset_url(link_key, args.hf_assets_repo))
        print(
            f"[{i}/{len(types)}] {t}: {link_key} -> {target}\n    "
            + "\n    ".join(f"{lbl}: {u}" for lbl, u in candidates),
            flush=True,
        )
        if args.dryrun:
            continue
        zip_path = target.parent / f"{target.name}.zip"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            download_any(candidates, zip_path)
            print(f"    extracting into {target.parent}", flush=True)
            extract(zip_path, target.parent)
            if not target.is_dir():
                raise RuntimeError(f"archive did not produce {target}; contents: {os.listdir(target.parent)}")
            marker.write_text(f"{link_key}\n{url}\n")
            print(f"    OK {t}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED {t}: {exc}", file=sys.stderr, flush=True)
            failed.append(t)
        finally:
            zip_path.unlink(missing_ok=True)
    if failed:
        print(f"ERROR: {len(failed)} asset type(s) failed: {failed} -- re-run to retry", file=sys.stderr)
        return 1
    print(f"done: {types} under {assets_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("datasets", help="LeRobot v2.1 demo datasets (one tar per task)")
    d.add_argument(
        "--box-links", required=True, help="robocasa/models/assets/box_links/box_links_ds.json of a robocasa clone"
    )
    d.add_argument("--dest", required=True, help="dataset base dir = robocasa DATASET_BASE_PATH/v1.0")
    d.add_argument("--split", default="target", choices=["target", "pretrain"])
    d.add_argument("--kind", default="atomic", choices=["atomic", "composite"])
    d.add_argument("--tasks", nargs="*", default=None, help="task names (default: every task of the split/kind)")
    d.add_argument("--source", default="auto", choices=SOURCES, help="auto = Box, then the HF mirror (default)")
    d.add_argument(
        "--hf-datasets-repo",
        default=HF_DATASETS_REPO,
        help=f"HF dataset repo mirroring the Box tree (default {HF_DATASETS_REPO})",
    )
    d.add_argument("--dryrun", action="store_true")
    d.set_defaults(fn=cmd_datasets)

    a = sub.add_parser("assets", help="kitchen assets (zips) into <assets-dir>")
    a.add_argument("--box-links", required=True, help="robocasa/models/assets/box_links/box_links_assets.json")
    a.add_argument("--assets-dir", required=True, help="<robocasa clone>/robocasa/models/assets")
    a.add_argument("--types", nargs="+", default=["all"], choices=list(ASSET_TYPES) + ["all"])
    a.add_argument("--source", default="auto", choices=SOURCES, help="auto = Box, then the HF mirror (default)")
    a.add_argument(
        "--hf-assets-repo",
        default=HF_ASSETS_REPO,
        help=f"HF dataset repo holding the 6 asset zips (default {HF_ASSETS_REPO})",
    )
    a.add_argument("--dryrun", action="store_true")
    a.set_defaults(fn=cmd_assets)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
