"""Show which EGL device MuJoCo would render on in this job, and how fast it renders there.

Run inside the allocation you would evaluate in, with the LIBERO env (takes a few seconds, no model/LIBERO needed):

    sr 1 48 ~/anaconda3/envs/libero-eval/bin/python examples/libero_v3/egl_probe.py [--allow_cpu_render] [--device N]

Prints the EGL device table (vendor, CUDA id, init result), the SLURM GPU hint, the GL_RENDERER string of the context
MuJoCo creates ("NVIDIA ..." = GPU, "llvmpipe" = CPU) and the offscreen frame rate on a toy scene.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from main_v3 import gl_renderer_string, select_egl_device  # noqa: E402

TOY_XML = """
<mujoco>
  <visual><global offwidth="512" offheight="512"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" castshadow="true"/>
    <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <camera name="cam" pos="1.5 -1.5 1.2" xyaxes="1 1 0 -0.4 0.4 1"/>
    <body pos="0 0 0.3"><joint type="free"/><geom type="box" size="0.1 0.1 0.1" rgba="0.9 0.2 0.2 1"/></body>
    <body pos="0.3 0.2 0.5"><joint type="free"/><geom type="sphere" size="0.08" rgba="0.2 0.4 0.9 1"/></body>
  </worldbody>
</mujoco>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow_cpu_render", action="store_true", help="also benchmark when only the software device opens")
    ap.add_argument("--device", type=int, default=-1, help="skip the probe and use this EGL device index")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--resolution", type=int, default=256)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    for var in ("SLURMD_NODENAME", "SLURM_JOB_ID", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS", "GPU_DEVICE_ORDINAL", "CUDA_VISIBLE_DEVICES"):
        logging.info(f"{var}={os.environ.get(var)!r}")

    if args.device < 0:
        chosen, table = select_egl_device(allow_software=args.allow_cpu_render)
        if chosen is None:
            logging.error(f"no usable EGL device ({len(table)} enumerated)")
            sys.exit(1)
        device = chosen.index
    else:
        device = args.device
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device)
    os.environ["EGL_DEVICE_ID"] = str(device)
    logging.info(f"MUJOCO_EGL_DEVICE_ID={device}")

    import mujoco  # noqa: E402  (imports mujoco.egl -> creates the display on MUJOCO_EGL_DEVICE_ID)

    model = mujoco.MjModel.from_xml_string(TOY_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.resolution, args.resolution)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="cam")
    frame = renderer.render()
    logging.info(f"GL renderer: {gl_renderer_string()}  (frame {frame.shape}, mean {frame.mean():.1f})")
    t0 = time.perf_counter()
    for _ in range(args.frames):
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera="cam")
        renderer.render()
    dt = time.perf_counter() - t0
    logging.info(f"{args.frames} frames at {args.resolution}x{args.resolution}: {args.frames / dt:.0f} fps ({1e3 * dt / args.frames:.1f} ms/frame)")


if __name__ == "__main__":
    main()
