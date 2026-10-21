"""Closed-loop LIBERO evaluation client for policies trained on NVIDIA LIBERO_LeRobot_v3 (cosmos_hs11 recipes).

Same loop as ``examples/libero/main.py`` (openpi), with the v3 conventions and the cosmos_hs11 reporting format:
* images are rotated 180 degrees (sim render -> upright, as stored in the v3 videos) and resized to 224 with padding;
* state = eef pos (3), eef axis-angle (3), gripper qpos (2) -- identical to ``observation.state`` of the dataset;
* the policy returns 7-D v3 actions; the gripper is mapped from [0, 1] (1 = open) to the env's [-1, 1] (binarized);
* ``summary.json`` with per-task / per-episode results (the format of the cosmos_hs11 ``closed_loop_eval.py``).

Runs in the LIBERO python env (``libero-eval``) with ``openpi_client`` on PYTHONPATH; the policy server is
``scripts/serve_policy.py policy:checkpoint --policy.config <config> --policy.dir <ckpt>/<step>`` in the openpi env.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import json
import logging
import math
import os
import pathlib
import time

import numpy as np

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def gripper_to_env(g: float, mode: str) -> float:
    """v3 convention ([0, 1], 1 = open) -> env ([-1, 1], -1 = open); ``passthrough`` for policies trained on env actions
    (e.g. openpi's official pi05_libero checkpoint, whose gripper is already in {-1, +1})."""
    if mode == "binarize":
        return -1.0 if g > 0.5 else 1.0
    if mode == "continuous":
        return float(np.clip(-(2.0 * g - 1.0), -1.0, 1.0))
    if mode == "passthrough":
        return float(np.clip(g, -1.0, 1.0))
    raise ValueError(mode)


EGL_EXTENSIONS = 0x3055
EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_CUDA_DEVICE_NV = 0x323A
MUJOCO_MAX_EGL_DEVICES = 10  # mujoco.egl enumerates at most 10 devices, so MUJOCO_EGL_DEVICE_ID must be < 10


@dataclasses.dataclass
class EglDevice:
    index: int
    extensions: str
    cuda_device: int | None = None  # EGL_CUDA_DEVICE_NV (NVIDIA devices only)
    initialized: bool | None = None  # None = not probed
    error: str = ""

    @property
    def vendor(self) -> str:
        if "EGL_NV_device_cuda" in self.extensions:
            return "nvidia"
        if "EGL_MESA_device_software" in self.extensions:
            return "software"
        return "other"

    def describe(self) -> str:
        state = "not probed" if self.initialized is None else ("init OK" if self.initialized else f"init FAILED {self.error}")
        cuda = f" cuda={self.cuda_device}" if self.cuda_device is not None else ""
        return f"EGL device {self.index}: {self.vendor}{cuda} [{self.extensions or 'no device extensions'}] {state}"


def slurm_gpu_hints() -> list[int]:
    """Physical GPU indices of this SLURM step (the cgroup only lets us open these /dev/nvidia* nodes); CUDA renumbers
    them to 0.. but the NVIDIA EGL device list keeps the physical order, so they are the natural first candidates."""
    for var in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS", "GPU_DEVICE_ORDINAL"):
        value = os.environ.get(var, "")
        try:
            hints = [int(x) for x in value.split(",") if x.strip()]
        except ValueError:
            continue
        if hints:
            return hints
    return []


def select_egl_device(max_devices: int = 64, allow_software: bool = False) -> tuple[EglDevice | None, list[EglDevice]]:
    """Pick the EGL device MuJoCo should render on inside this job, and return (chosen, all devices probed).

    Inside a SLURM allocation the EGL device list is NOT ordered like CUDA_VISIBLE_DEVICES and non-allocated GPUs fail
    to initialize at the driver level, so neither "device 0" nor the CUDA index is reliable. Candidates are tried in
    this order: NVIDIA devices at the physical index SLURM reports, other NVIDIA devices, other hardware devices, and
    Mesa's software rasterizer (llvmpipe) last. The software device always initializes but renders LIBERO ~5x slower
    than the GPU, so it is only accepted with ``allow_software`` (the caller should fail loudly otherwise).
    Same ctypes wrappers as mujoco/egl/egl_ext.py, without importing mujoco.egl (it creates the display at import).
    """
    import ctypes

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        from OpenGL import EGL
        from OpenGL import error as gl_error
        from OpenGL.platform import ctypesloader

        try:
            ctypesloader.loadLibrary(ctypes.cdll, "OpenGL", mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

        def proc(name: str, proto):
            addr = EGL.eglGetProcAddress(name)
            return proto(addr) if addr else None

        query = proc("eglQueryDevicesEXT", ctypes.CFUNCTYPE(EGL.EGLBoolean, EGL.EGLint, ctypes.POINTER(EGL.EGLDeviceEXT), ctypes.POINTER(EGL.EGLint)))
        get_display = proc("eglGetPlatformDisplayEXT", ctypes.CFUNCTYPE(EGL.EGLDisplay, EGL.EGLenum, ctypes.c_void_p, ctypes.POINTER(EGL.EGLint)))
        query_string = proc("eglQueryDeviceStringEXT", ctypes.CFUNCTYPE(ctypes.c_char_p, EGL.EGLDeviceEXT, EGL.EGLint))
        query_attrib = proc("eglQueryDeviceAttribEXT", ctypes.CFUNCTYPE(EGL.EGLBoolean, EGL.EGLDeviceEXT, EGL.EGLint, ctypes.POINTER(ctypes.c_ssize_t)))
        if query is None or get_display is None:
            raise RuntimeError("EGL_EXT_device_enumeration / EGL_EXT_platform_device not exported by libEGL")
    except Exception as e:  # noqa: BLE001 -- no usable EGL at all on this machine
        logging.warning(f"EGL device query unavailable ({type(e).__name__}: {e})")
        return None, []
    devices = (EGL.EGLDeviceEXT * max_devices)()
    num = EGL.EGLint()
    if query(max_devices, devices, num) != EGL.EGL_TRUE:
        logging.warning("eglQueryDevicesEXT failed")
        return None, []

    table: list[EglDevice] = []
    for i in range(num.value):
        ext = ""
        cuda = None
        if query_string is not None:
            try:
                raw = query_string(devices[i], EGL_EXTENSIONS)
                ext = raw.decode() if raw else ""
            except Exception:  # noqa: BLE001
                ext = ""
        if query_attrib is not None and "EGL_NV_device_cuda" in ext:
            value = ctypes.c_ssize_t()
            if query_attrib(devices[i], EGL_CUDA_DEVICE_NV, ctypes.byref(value)) == EGL.EGL_TRUE:
                cuda = int(value.value)
        EGL.eglGetError()  # clear any error latched by the raw queries (PyOpenGL checks it after its own calls)
        table.append(EglDevice(index=i, extensions=ext, cuda_device=cuda))

    def initialize(dev: EglDevice) -> bool:
        display = get_display(EGL_PLATFORM_DEVICE_EXT, devices[dev.index], None)
        if not display:  # NULL pointer
            dev.error = f"(eglGetPlatformDisplayEXT: 0x{EGL.eglGetError():x})"
            dev.initialized = False
            return False
        EGL.eglGetError()
        try:
            ok = bool(EGL.eglInitialize(display, None, None))
            err = EGL.eglGetError()
        except gl_error.GLError as e:
            ok, err = False, e.err
        dev.initialized = ok
        if ok:
            EGL.eglTerminate(display)
        else:
            dev.error = f"(eglInitialize: 0x{int(err or 0):x})"
        return ok

    hints = slurm_gpu_hints()
    rank = {"nvidia": 1, "other": 2, "software": 3}
    order = sorted(table, key=lambda d: (0 if (d.vendor == "nvidia" and d.index in hints) else rank[d.vendor], d.index))
    chosen: EglDevice | None = None
    for dev in order:
        if chosen is not None and dev.vendor != "nvidia":
            break  # a hardware device works; don't poke the DRM nodes (noisy "Permission denied" warnings) or llvmpipe
        if initialize(dev) and chosen is None:
            chosen = dev
    if hints:
        logging.info(f"SLURM physical GPU index hint: {hints} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
    for dev in table:
        logging.info(dev.describe())
    if chosen is not None and chosen.vendor == "software":
        msg = (f"only Mesa's software EGL device (index {chosen.index}) initializes in this job: no NVIDIA EGL device could be "
               "opened, see the device table above. CPU rendering is ~5x slower than the GPU (LIBERO episode ~100 s vs ~20 s).")
        if not allow_software:
            logging.error(msg + " Pass --allow_cpu_render (ALLOW_CPU_RENDER=1 in eval_libero_v3.sh) to run anyway.")
            return None, table
        logging.warning(msg)
    if chosen is not None and chosen.index >= MUJOCO_MAX_EGL_DEVICES:
        logging.warning(f"EGL device index {chosen.index} >= {MUJOCO_MAX_EGL_DEVICES}: mujoco.egl only enumerates 10 devices and will reject it")
    return chosen, table


def gl_renderer_string() -> str:
    """GL_VENDOR / GL_RENDERER of the context that is current on this thread ('' if none); 'llvmpipe' = CPU rendering."""
    try:
        from OpenGL import GL

        vendor, renderer = GL.glGetString(GL.GL_VENDOR), GL.glGetString(GL.GL_RENDERER)
        return f"{(vendor or b'').decode()} / {(renderer or b'').decode()}".strip(" /")
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {type(e).__name__}>"


def get_libero_env(task, resolution, seed, render_gpu_device_id=0):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution, render_gpu_device_id=render_gpu_device_id
    )
    env.seed(seed)
    return env, str(task.language)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--task_suite", default="libero_goal", choices=sorted(MAX_STEPS))
    ap.add_argument("--task_ids", default="all", help="comma-separated task ids or 'all'")
    ap.add_argument("--num_trials_per_task", type=int, default=50)
    ap.add_argument("--replan_steps", type=int, default=5, help="env steps executed per policy call (openpi default 5 of the 10-step chunk)")
    ap.add_argument("--resize_size", type=int, default=224)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gripper", default="binarize", choices=["binarize", "continuous", "passthrough"],
                    help="v3 [0,1] gripper -> env [-1,1] mapping; passthrough = policy already outputs env gripper (official openpi LIBERO ckpt)")
    ap.add_argument("--render_gpu_device_id", type=int, default=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "-1")),
                    help="EGL device index for MuJoCo rendering; -1 (default) probes for the first device that initializes in this job")
    ap.add_argument("--allow_cpu_render", action="store_true",
                    help="accept Mesa's software EGL device (llvmpipe) when no NVIDIA EGL device opens in this job (~5x slower)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--save_gifs", action="store_true")
    ap.add_argument("--gif_stride", type=int, default=2)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    # Headless MuJoCo rendering on the allocated GPU: probe the EGL device list for the one this job can open, then pin it
    # for mujoco AND robosuite (both require an explicit index once MUJOCO_EGL_DEVICE_ID is set).
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if args.render_gpu_device_id < 0:
        chosen, table = select_egl_device(allow_software=args.allow_cpu_render)
        if chosen is None:
            raise RuntimeError(f"no usable EGL device in this job ({len(table)} EGL devices enumerated, see the table above); "
                               "is a GPU allocated? Run examples/libero_v3/egl_probe.py inside the allocation to inspect.")
        logging.info(f"EGL: rendering on device index {chosen.index} ({chosen.vendor}) of {len(table)}")
        args.render_gpu_device_id = chosen.index
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.render_gpu_device_id)
    os.environ["EGL_DEVICE_ID"] = str(args.render_gpu_device_id)
    # robosuite asserts at import that MUJOCO_EGL_DEVICE_ID is one of CUDA_VISIBLE_DEVICES, but under SLURM cgroups CUDA
    # renumbers the allocation to "0" while the EGL list keeps physical order. This client does no CUDA work (EGL renders,
    # torch only unpickles init states), so align the CUDA variable with the EGL index for this process only.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and str(args.render_gpu_device_id) not in cvd.split(","):
        logging.info(f"CUDA_VISIBLE_DEVICES={cvd!r} -> {args.render_gpu_device_id} in the eval client (EGL index, robosuite import check)")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.render_gpu_device_id)

    from libero.libero import benchmark
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    np.random.seed(args.seed)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata() if hasattr(client, "get_server_metadata") else {}
    logging.info(f"server metadata: {metadata}")

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task_ids = list(range(suite.n_tasks)) if args.task_ids == "all" else [int(x) for x in args.task_ids.split(",")]
    max_steps = MAX_STEPS[args.task_suite]

    results = []
    total_ep, total_succ = 0, 0
    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.render_gpu_device_id)
        if task_id == task_ids[0]:  # ground truth of where MuJoCo renders (the env's GL context is current on this thread)
            renderer = gl_renderer_string()
            logging.info(f"GL renderer: {renderer}")
            if ("llvmpipe" in renderer or "softpipe" in renderer) and not args.allow_cpu_render:
                raise RuntimeError(f"MuJoCo is rendering on the CPU ({renderer}); pass --allow_cpu_render to accept the ~5x slowdown")
        episodes = []
        for ep in range(args.num_trials_per_task):
            t_start = time.monotonic()
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            plan = collections.deque()
            frames = []
            success, t, error = False, 0, None
            logging.info(f"[{args.task_suite} t{task_id} ep{ep}] start: {description}")
            try:
                while t < max_steps + args.num_steps_wait:
                    if t < args.num_steps_wait:
                        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size))
                    if args.save_gifs and (t - args.num_steps_wait) % args.gif_stride == 0:
                        frames.append(img)
                    if not plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist,
                            "observation/state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ).astype(np.float32),
                            "prompt": description,
                        }
                        chunk = np.asarray(client.infer(element)["actions"])
                        if len(chunk) < args.replan_steps:
                            raise RuntimeError(f"policy returned {len(chunk)} steps < replan_steps {args.replan_steps}")
                        plan.extend(chunk[: args.replan_steps])
                    a = np.asarray(plan.popleft(), dtype=np.float64)
                    env_action = [float(x) for x in a[:6]] + [gripper_to_env(float(a[6]), args.gripper)]
                    obs, _, done, _ = env.step(env_action)
                    t += 1
                    if (t - args.num_steps_wait) % 100 == 0:
                        logging.info(f"[{args.task_suite} t{task_id} ep{ep}] step {t - args.num_steps_wait}/{max_steps}")
                    if done:
                        success = True
                        break
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                logging.error(f"task {task_id} episode {ep}: {error}")
            steps = t - args.num_steps_wait
            episodes.append({"episode": ep, "success": bool(success), "steps": int(steps), "error": error, "elapsed_s": round(time.monotonic() - t_start, 3)})
            total_ep += 1
            total_succ += int(success)
            logging.info(f"[{args.task_suite} t{task_id} ep{ep}] success={success} steps={steps} | running SR {total_succ}/{total_ep}")
            if args.save_gifs and frames:
                try:
                    import imageio

                    gif_dir = out_dir / "gifs" / f"task_{task_id:03d}"
                    gif_dir.mkdir(parents=True, exist_ok=True)
                    imageio.mimsave(gif_dir / f"episode_{ep:03d}.gif", frames, duration=args.gif_stride / 20.0)
                except Exception as e:  # noqa: BLE001
                    logging.warning(f"gif export failed: {e}")
        env.close()
        n_succ = sum(e["success"] for e in episodes)
        results.append(
            {
                "task_id": task_id,
                "task_description": description,
                "episodes": len(episodes),
                "successes": n_succ,
                "success_rate": n_succ / max(1, len(episodes)),
                "episode_results": episodes,
            }
        )
        summary = {
            "task_suite": args.task_suite,
            "total_episodes": total_ep,
            "total_successes": total_succ,
            "overall_success_rate": total_succ / max(1, total_ep),
            "num_trials_per_task": args.num_trials_per_task,
            "selected_task_ids": task_ids,
            "action_space": "libero_v3",
            "gripper_mapping": args.gripper,
            "replan_steps": args.replan_steps,
            "server_metadata": metadata,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "task_results": results,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logging.info(f"DONE {args.task_suite}: SR {total_succ}/{total_ep} = {total_succ / max(1, total_ep):.3f} -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
