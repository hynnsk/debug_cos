# Vendored V-JEPA 2.1 encoder

Source: https://github.com/facebookresearch/vjepa2 at commit `204698b45b3712590f06245fbfba32d3be539812` (cloned 2026-09-17).

| File | Origin | Changes |
| --- | --- | --- |
| `vision_transformer.py` | `app/vjepa_2_1/models/vision_transformer.py` | relative imports, `apply_masks` inlined, `trunc_normal_` from `torch.nn.init` |
| `modules.py` | `app/vjepa_2_1/models/utils/modules.py` | local `drop_path` (no timm import), `torch.backends.cuda.sdp_kernel()` -> `contextlib.nullcontext()` |
| `patch_embed.py` | `app/vjepa_2_1/models/utils/patch_embed.py` | none |

The upstream hub loaders (`src/hub/backbones.py::_make_vjepa2_1_model`) build the encoder with
`patch_size=16, tubelet_size=2, use_rope=True, use_sdpa=True, uniform_power=False, img_temporal_dim_size=1,
interpolate_rope=True` and load the `ema_encoder` entry of the released checkpoints with `strict=True`
after stripping the `module.` / `backbone.` prefixes. `cosmos_framework/model/generator/repa/vjepa_teacher.py`
reproduces exactly that construction.

License: MIT (see `LICENSE`, copied from the upstream repository root).
