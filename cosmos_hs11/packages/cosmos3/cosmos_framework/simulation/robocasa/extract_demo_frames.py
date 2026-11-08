# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""cosmos_hs11: dump frame 0 of chosen demo episodes (v2.1 videos, PyAV; cosmos3-pt env) for replay_consistency_check.py.

python cosmos_framework/simulation/robocasa/extract_demo_frames.py <ROBOCASA_V21_ROOT> <out_dir> '{"Task": [0, 100]}'
"""

import glob
import json
import sys

import av
import numpy as np
from PIL import Image

root, out = sys.argv[1], sys.argv[2]
plan = json.loads(sys.argv[3])  # {task: [ep, ...]}
for task, eps in plan.items():
    ds = glob.glob(f"{root}/{task}/*/lerobot")[0]
    for ep in eps:
        for cam in ("robot0_agentview_left", "robot0_eye_in_hand"):
            vids = glob.glob(f"{ds}/videos/**/observation.images.{cam}/episode_{ep:06d}.mp4", recursive=True)
            assert vids, (task, ep, cam)
            c = av.open(vids[0])
            frames = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
            np.save(f"{out}/{task}_ep{ep}_{cam}.npy", np.stack(frames[:3]))  # first 3 frames
            Image.fromarray(frames[0]).save(f"{out}/{task}_ep{ep}_{cam}_f0.png")
        print(task, ep, "frames", len(frames), "saved")
