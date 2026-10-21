"""Serve a trained checkpoint with the data pipeline / model described by its training TOML (cosmos_hs11 style).

    python scripts/hs11/serve_policy_toml.py --toml scripts/hs11/toml/libero10_3ep_base_v1.toml \
        --dir checkpoints/pi05_libero_v3_libero_10_3ep_base/libero10_3ep_base_v1/5000 [--port 8000] [--check] [section.key=value ...]

Unlike ``scripts/serve_policy.py --policy.config <name>`` this honours model-level TOML changes (e.g. [model].action_horizon),
so evaluation always matches training. Weights come from ``<dir>/params``, norm stats from ``<dir>/assets/<suite>``.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import socket
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from openpi.hs11 import toml_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toml", required=True)
    ap.add_argument("--dir", required=True, help="checkpoint step dir holding params/ and assets/")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--check", action="store_true", help="only resolve the config and verify the checkpoint layout")
    ap.add_argument("--guidance", type=float, default=1.0, help="classifier-free guidance on the prompt: v = v_u + g (v_c - v_u); 1.0 = off")
    ap.add_argument("--uncond-prompt", default="", help="prompt of the unconditional branch (default: empty prompt)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    # The init checkpoint only matters for training; force "base" so a deleted Reptile dir cannot block serving.
    config = toml_config.load_train_config(args.toml, [*args.overrides, "checkpoint.init=base", "wandb.enabled=false"])
    ckpt = pathlib.Path(args.dir)
    data_config = config.data.create(config.assets_dirs, config.model)
    stats = ckpt / "assets" / str(data_config.asset_id) / "norm_stats.json"
    print(f"config {config.name} (exp {config.exp_name}) | suite {config.data.suite} | horizon {config.model.action_horizon} | ckpt {ckpt}")
    if not (ckpt / "params").is_dir():
        raise FileNotFoundError(f"{ckpt / 'params'} not found (pass the <step> directory of a training checkpoint)")
    if not stats.is_file():
        raise FileNotFoundError(f"{stats} not found (norm stats are copied into every training checkpoint)")
    if args.check:
        print("check OK")
        return

    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server

    sample_kwargs = None
    if args.guidance != 1.0:
        import jax.numpy as jnp
        import numpy as np

        from openpi.models import tokenizer as _tokenizer

        if getattr(config.model, "discrete_state_input", False):
            raise NotImplementedError("CFG needs the continuous-state prompt format (discrete_state_input=False)")
        tokens, mask = _tokenizer.PaligemmaTokenizer(config.model.max_token_len).tokenize(args.uncond_prompt)
        sample_kwargs = {
            "guidance": float(args.guidance),
            "uncond_tokenized_prompt": jnp.asarray(np.asarray(tokens)[None]),
            "uncond_tokenized_prompt_mask": jnp.asarray(np.asarray(mask)[None]),
        }
        logging.info("classifier-free guidance %.2f with unconditional prompt %r", args.guidance, args.uncond_prompt)
    policy = _policy_config.create_trained_policy(config, ckpt, sample_kwargs=sample_kwargs)
    # Warm up: the first sample_actions call JIT-compiles for minutes; do it BEFORE listening so no client ever waits on it.
    import time

    import numpy as np

    warm = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": "warm up",
    }
    t0 = time.monotonic()
    actions = np.asarray(policy.infer(warm)["actions"])
    logging.info("warm-up inference done in %.1fs (compile); actions %s", time.monotonic() - t0, actions.shape)
    t0 = time.monotonic()
    policy.infer(warm)
    logging.info("steady-state inference %.0f ms", 1000 * (time.monotonic() - t0))
    logging.info("serving on %s:%d (ip %s)", socket.gethostname(), args.port, socket.gethostbyname(socket.gethostname()))
    metadata = {**(policy.metadata or {}), "guidance": float(args.guidance), "uncond_prompt": args.uncond_prompt}
    websocket_policy_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
