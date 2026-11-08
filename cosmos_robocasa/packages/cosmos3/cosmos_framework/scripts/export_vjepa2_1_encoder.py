# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Export the ``ema_encoder`` of a V-JEPA 2.1 release checkpoint into a small encoder-only file.

The released files also carry the online encoder, the predictor and the optimizer state (1.7 GB for ViT-B,
5.2 GB for ViT-L); every training rank mmaps them at start-up. This writes ``<stem>.ema_encoder.pt`` next to
the original (cleaned keys, fp32), which ``VJEPA21Teacher`` picks up automatically when present.

Usage::

    python -m cosmos_framework.scripts.export_vjepa2_1_encoder \
        --checkpoint $COSMOS_STORAGE/checkpoints/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cosmos_framework.model.generator.repa.vjepa_teacher import (
    VJEPA21_TEACHERS,
    build_vjepa21_encoder,
    load_encoder_state_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Release checkpoint (.pt) with an 'ema_encoder' entry")
    parser.add_argument("--output", type=Path, default=None, help="Default: <checkpoint stem>.ema_encoder.pt")
    parser.add_argument("--key", default="ema_encoder", help="Checkpoint entry to export (ema_encoder | encoder)")
    parser.add_argument("--verify", action="store_true", help="Instantiate the matching encoder and load strictly")
    args = parser.parse_args()

    state_dict = load_encoder_state_dict(args.checkpoint, checkpoint_key=args.key)
    output = args.output or args.checkpoint.with_name(args.checkpoint.name.replace(".pt", f".{args.key}.pt"))
    if args.verify:
        spec = next((s for s in VJEPA21_TEACHERS.values() if s.filename == args.checkpoint.name), None)
        if spec is None:
            raise SystemExit(f"--verify needs a known release file name, got {args.checkpoint.name}")
        encoder = build_vjepa21_encoder(spec)
        encoder.load_state_dict(state_dict, strict=True)
        print(f"verified strict load into {spec.arch} ({sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M params)")
    torch.save({k: v.contiguous() for k, v in state_dict.items()}, output)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.0f} MB, {len(state_dict)} tensors)")


if __name__ == "__main__":
    main()
