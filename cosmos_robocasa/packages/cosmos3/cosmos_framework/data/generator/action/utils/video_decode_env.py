# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Make the CUDA build of torchcodec loadable without a hand-crafted ``LD_LIBRARY_PATH``.

``lerobot.datasets.video_utils.decode_video_frames`` prefers torchcodec, whose ``libtorchcodec_core*.so``
links against NVIDIA NPP (``libnppicc.so.13``, ...). In a pip/conda environment those libraries live in
the ``nvidia-*-cu13`` wheels (``site-packages/nvidia/cu13/lib``), which the dynamic loader does not search
by default, so every video decode fails with ``Could not load libtorchcodec ... libnppicc.so.13`` unless
the shell exported ``LD_LIBRARY_PATH`` first. :func:`ensure_video_decoder_libs` pre-loads the NPP
libraries with ``RTLD_GLOBAL`` (the same trick ``torch`` uses for cuDNN / nvrtc), so later ``dlopen`` calls
resolve them from the already-loaded set. Idempotent and a no-op when the libraries are already
resolvable; DataLoader workers forked from a process that called it inherit the mapping.
"""

from __future__ import annotations

import ctypes
import glob
import os

_NPP_LIB_ORDER = (
    "libnppc",  # NPP core: every other NPP library depends on it
    "libnppicc",
    "libnppig",
    "libnppial",
    "libnppidei",
    "libnppif",
    "libnppim",
    "libnppist",
    "libnppisu",
    "libnppitc",
)
_DONE = False


def _candidate_lib_dirs() -> list[str]:
    dirs: list[str] = []
    try:
        import nvidia  # namespace package of the nvidia-*-cu1x wheels
    except ImportError:
        return dirs
    for base in getattr(nvidia, "__path__", []):
        dirs.extend(sorted(glob.glob(os.path.join(base, "*", "lib"))))
    return [d for d in dirs if os.path.isdir(d)]


def ensure_video_decoder_libs() -> bool:
    """Pre-load NVIDIA NPP shared libraries. Returns True when ``libnppicc`` is resolvable afterwards."""
    global _DONE
    if _DONE:
        return True
    try:
        ctypes.CDLL("libnppicc.so.13", mode=ctypes.RTLD_GLOBAL)
        _DONE = True
        return True
    except OSError:
        pass
    loaded_any = False
    for lib_dir in _candidate_lib_dirs():
        for name in _NPP_LIB_ORDER:
            for path in sorted(glob.glob(os.path.join(lib_dir, f"{name}.so*"))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    loaded_any = True
                except OSError:
                    continue
    try:
        ctypes.CDLL("libnppicc.so.13", mode=ctypes.RTLD_GLOBAL)
        _DONE = True
        return True
    except OSError:
        _DONE = loaded_any
        return False
