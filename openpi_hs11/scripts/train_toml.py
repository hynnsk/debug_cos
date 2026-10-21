"""Run openpi's ``scripts/train.py`` from a TOML experiment file (cosmos_hs11 style).

    python scripts/train_toml.py --toml scripts/hs11/toml/libero_goal_3ep_base_v1.toml [--dryrun] [section.key=value ...]

Overrides are applied on the TOML before the config is built, e.g. ``train.num_train_steps=20 wandb.enabled=false``.
``--dryrun`` prints the resolved config, builds the dataset (CPU is fine) and exits.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root -> ``scripts`` importable

from openpi.hs11 import toml_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toml", required=True)
    ap.add_argument("--dryrun", action="store_true", help="resolve the config, build the dataset, print, exit")
    ap.add_argument("overrides", nargs="*", help="section.key=value overrides")
    args = ap.parse_args()

    config = toml_config.load_train_config(args.toml, args.overrides)
    print(f"=== {args.toml} -> TrainConfig {config.name} / exp {config.exp_name}")
    print(toml_config.describe(config))
    print(toml_config.checkpoint_status(config.checkpoint_dir, overwrite=config.overwrite, resume=config.resume))
    if args.dryrun:
        import openpi.training.data_loader as _data_loader

        data_config = config.data.create(config.assets_dirs, config.model)
        dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
        print(f"dataset: {len(dataset)} frames, norm_stats={'yes' if data_config.norm_stats is not None else 'MISSING'}, "
              f"asset_id={data_config.asset_id}, checkpoint_dir={config.checkpoint_dir}")
        return
    from scripts import train

    logging.basicConfig(level=logging.INFO)
    train.main(config)


if __name__ == "__main__":
    main()
