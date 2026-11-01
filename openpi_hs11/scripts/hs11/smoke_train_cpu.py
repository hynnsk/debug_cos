"""CPU smoke of ``scripts/train.py`` on a real LIBERO_v3 recipe with the DUMMY pi05 model (no GPU, no pi05_base needed).

Exercises the whole openpi training path with the hs11 dataset: spawned DataLoader workers decoding the real videos,
norm stats, transforms, train_step, checkpoint + assets. Must be a real file (not ``python -``) because spawned workers
re-import ``__main__``::

    JAX_PLATFORMS=cpu .venv/bin/python scripts/hs11/smoke_train_cpu.py --config pi05_libero_v3_libero_goal_3ep_base \
        --out $COSMOS_STORAGE/smoke_openpi_hs11 --num-workers 2
    ... --language-ranking        # also exercises the language-ranking negative stream + metrics
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import time

import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# repo root on sys.path so ``from scripts import train`` works when this file is run directly
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import tyro

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.weight_loaders as wl


@dataclasses.dataclass
class Args:
    config: str = "pi05_libero_v3_libero_goal_3ep_base"
    out: str = "./checkpoints_smoke"
    num_workers: int = 2
    num_train_steps: int = 3
    batch_size: int = 4
    # cosmos_hs11 language ranking (NegativePrompt -> TokenizeNegativePrompt -> paired hinge in pi0.compute_loss).
    language_ranking: bool = False


def main(args: Args) -> None:
    from scripts import train

    out = pathlib.Path(args.out)
    cfg = _config.get_config(args.config)
    cfg = dataclasses.replace(
        cfg,
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_horizon=10,
            discrete_state_input=False,
            language_ranking=pi0_config.LanguageRankingConfig(
                enabled=args.language_ranking, loss_weight=0.022, margin=0.1, raw_action_dim=7
            ),
        ),
        weight_loader=wl.NoOpWeightLoader(),
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        log_interval=1,
        save_interval=2,
        keep_period=1,
        num_workers=args.num_workers,
        exp_name="cpu_smoke",
        checkpoint_base_dir=str(out),
        wandb_enabled=False,
        overwrite=True,
    )
    t0 = time.time()
    print("START train.main", flush=True)
    train.main(cfg)
    print(f"train.main returned after {time.time() - t0:.0f}s", flush=True)
    ck = out / cfg.name / "cpu_smoke"
    print("checkpoints:", sorted(p.name for p in ck.iterdir() if p.is_dir()), flush=True)
    print("assets saved:", sorted(str(p.relative_to(ck)) for p in ck.rglob("norm_stats.json")), flush=True)
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
