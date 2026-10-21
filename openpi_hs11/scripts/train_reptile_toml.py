"""Run ``scripts/train_reptile.py`` from a TOML experiment file (cosmos_hs11 style).

    python scripts/train_reptile_toml.py --toml scripts/hs11/toml/reptile_pi05_v1.toml [--dryrun] [section.key=value ...]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from openpi.hs11 import toml_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toml", required=True)
    ap.add_argument("--dryrun", action="store_true", help="resolve the config, print, exit")
    ap.add_argument("overrides", nargs="*", help="section.key=value overrides")
    args = ap.parse_args()

    config = toml_config.load_reptile_config(args.toml, args.overrides)
    print(f"=== {args.toml} -> ReptileConfig exp {config.exp_name}")
    print(toml_config.describe(config))
    print(toml_config.checkpoint_status(config.checkpoint_dir, overwrite=config.overwrite, resume=config.resume))
    if args.dryrun:
        print(f"support {config.sampler.support_size} / query {config.sampler.query_size} samples per meta-episode, "
              f"{config.sampler.support_size // config.inner_batch_size} micro-batches per inner step, checkpoint_dir={config.checkpoint_dir}")
        return
    from scripts import train_reptile

    train_reptile.main(config)


if __name__ == "__main__":
    main()
