# V-JEPA 2.1 representation alignment (REPA) for Cosmos3-Edge LIBERO-10 post-training

`cosmos_hs10` = `cosmos_hs08` (mid-trained Cosmos3-Edge, full `moe_gen` post-training on the fixed 30-demo
LIBERO-10 subset with held-out validation) **plus one extra loss**: the video tokens of an intermediate MoT
decoder block are regressed, through a small MLP, onto frozen V-JEPA 2.1 features of the frames the model is
asked to predict, REPA-style (Yu et al., *Representation Alignment for Generation*, 2024).

```
loss_total = 10 * fm_vision + 10 * fm_action                     (unchanged hs08 recipe)
           + repa.loss_weight * ( 1 - cos( proj(h_k), adapter(V-JEPA-2.1(frames)) ) )  # default weight 0.5; proj = MLP (default) | Linear (v4)
```

| Piece | Path |
| --- | --- |
| Frozen teacher (V-JEPA 2.1 `ema_encoder`, vendored encoder code) | `cosmos_framework/model/generator/repa/vjepa_teacher.py`, `.../repa/vjepa2_1/` |
| REPA projector + target adapters (3 variants) | `cosmos_framework/model/generator/repa/adapters.py` |
| Network glue (token gather, view concat, loss inputs) | `cosmos_framework/model/generator/repa/alignment.py` (`RepaAlignmentHead`, registered as `net.repa_head`) |
| Hidden-state capture in the MoT layer loop | `cosmos_framework/model/generator/mot/unified_mot.py::_impl_forward` |
| Net forward outputs `repa_pred` / `repa_target` | `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py` |
| Teacher call + loss term | `cosmos_framework/model/generator/omni_mot_model.py` (`_compute_repa_teacher_tokens`, `_compute_losses`) |
| Config schema | `configs/base/defaults/model_config.py::RepaConfig` (Hydra) and `configs/toml_config/sft_config.py::RepaTomlConfig` (`[model.repa]`) |
| Experiment / TOMLs / launcher | `action_policy_libero_edge_repa.py`, `examples/toml/sft_config/action_policy_libero_10_edge_repa*.toml`, `examples/launch_sft_action_policy_libero_10_edge_repa.sh` |
| Tests | `cosmos_framework/model/generator/repa/*_test.py`, `configs/toml_config/repa_toml_test.py` |

## 1. What is aligned with what

**Student side (MoT).** LIBERO-10 samples are 17 frames (frame 0 = current observation, frames 1..16 =
future) of the `concat_view` canvas (third-person | wrist, 256x512). The Wan2.2 VAE (4x16x16) + 2x2 patchify give
**5 latent frames x 5 x 10 video tokens** per sample: the 256x512 canvas is snapped to 192x320 with 160 px of
content height, and the reflection-padded latent rows are cropped before packing (`_remove_padding_from_latent`),
so each view is a **5x5** token grid and the predicted (noised) part is **4 x 5 x 10** tokens (latent frames
1..4; latent frame 0 is the clean conditioning frame and is never aligned). The hidden state read is the residual
stream after `repa.layer_index` decoder blocks (default **8 of 28**; second candidate **14**), captured in the
eager layer loop of `_impl_forward` (outside the per-block `torch.compile` / activation-checkpoint wrappers), then
projected by the student-side projector: the REPA MLP (`Linear-SiLU-Linear-SiLU-Linear`, hidden 2048, out = teacher
dim; `projector_type = "mlp"`, base / v2 / v3) or, in the v4 recipe, a single `Linear(2048, D_t)`
(`projector_type = "linear"`), which removes the projector's own capacity so `h_k` itself has to line up with the
teacher features up to an affine map.

**Token-relation distillation (v5, VideoREPA-style).** Instead of pulling each projected student token onto its
teacher token, `repa_rel_loss` matches the *geometry among the tokens of one video*: for every sample the
`n x n` pairwise cosine-similarity maps `R(x) = normalize(x) normalize(x)^T` of the projected student tokens and of
the adapted teacher tokens (`n = 4 x 5 x 10 = 200` predicted tokens per LIBERO window, so spatial pairs within a
frame and temporal pairs across frames are covered by one map) are compared entry-wise with the squared
(`relation_distance = "l2"`) or absolute (`"l1"`) difference, averaged over all entries of all samples. The term is
invariant to any rotation / per-token rescaling of either feature space, so it constrains relations, not absolute
directions. Both `repa_loss` (`1 - cos`) and `repa_rel_loss` are always computed and logged; the weights
`loss_weight` / `relation_loss_weight` decide what is trained (`*_v5.toml`: `0.0` / `5.0`, parameter-free `avgpool`
target, MLP projector). Cost: two `[B, 200, 200]` fp32 matmuls per step, negligible.

**Transition alignment (v8).** `objective = "temporal_difference"` compares the same spatial patch across adjacent
latent frames: `MLP(h[t+1,p]) - MLP(h[t,p])` against `y[t+1,p] - y[t,p]`. A pair is used only when the packed frame
indexes differ by exactly one, so a partially noised sequence never creates a synthetic transition across a gap.
The trained cosine is logged as `repa_cos_sim_transition`; `repa_cos_sim` remains the raw absolute-feature diagnostic.

**Visual-token SIGReg (v9).** The base REPA loss is retained and `[model.sigreg]` adds an Epps--Pulley SIGReg term to
the MoT visual rows only (text, action, sound, and LiDAR rows are excluded). `layer_index` selects block 8 or 28;
the default recipe uses block 8, 256 random slices, 17 quadrature points over `[-5,5]`, and weight `0.1`. Since the
vision and action flow-matching losses are each scaled by 10, this is an effective relative coefficient of `0.01`
under the corresponding unscaled-loss convention. Random
directions are synchronized per iteration, the empirical characteristic function is reduced across both GPUs, and
slice chunks are activation-checkpointed to avoid materializing `[tokens, slices, points]`.

**Spatially normalized alignment (v10).** `objective = "spatial_normalized"` independently transforms both sides as
`(x[t,p] - mean_p(x[t,p])) / (std_p(x[t,p]) + eps)` for every frame and feature channel before token-wise cosine.
No statistic crosses frames or samples. The trained cosine is logged as `repa_cos_sim_spatial_norm`.

**Teacher side (V-JEPA 2.1).** The dataset ships the un-resized uint8 frames as `data_batch["video_native"]`
(`keep_native_video=True`). Frames 1..16 of **each camera view** (256x256, the native LIBERO resolution) go
through the frozen `ema_encoder` independently: ImageNet mean/std normalization, 2-frame tubelets, 16x16
patches -> **8 x 16 x 16 tokens per view** (D=768 for ViT-B/16, 1024 for ViT-L/16). A 256 px input is the
V-JEPA 2.1 RoPE base grid (`interpolate_rope=True` rescales positions onto the 16x16 pretraining grid), so no
resolution adaptation is needed; `teacher_input_size=384` is possible but 2.3x more tokens.

**Target adapters (`repa.target_adapter`).** Per view the teacher grid is brought onto the view's token grid
`(4, 5, 5)`; the two views are then concatenated along width -> `(4, 5, 10)`, matching the canvas layout, and the
predicted frames are selected (frame index = latent frame - 1; temporally, latent frame t covers raw frames
4t-3..4t = tubelets 2(t-1), 2(t-1)+1, so pairing tubelets is exact).

| variant | `target_adapter` | what it does | learnable params (ViT-B) |
| --- | --- | --- | --- |
| 1 | `avgpool` | `adaptive_avg_pool3d`: tubelet pairs averaged in time, adaptive average pooling 16->5 in space | 0 |
| 2 | `avgpool_conv` | variant 1 -> depthwise `Conv3d(3^3)` -> `1x1x1 Conv3d`; **identity init** (== variant 1 at step 0) | 0.6 M |
| 3 | `strided_conv` | strided depthwise `Conv3d` with kernel/stride derived from `8x16x16 -> 4x5x5` (windows == pooling bins) -> `1x1x1 Conv3d`; **box-average/identity init** (== variant 1 at step 0); `target_adapter_depthwise=false` = full conv | 0.6 M (19 M full) |

The teacher is always under `no_grad` (stop gradient). In variants 2/3 the adapter *is* trained by the loss, so the
target can drift; watch `repa_cos_sim` (a value racing to 1.0 much faster than in variant 1 suggests the adapter is
collapsing the target). `repa_loss = 1 - mean cos` is logged in training (wandb picks up every `*loss*` key) and
`val/repa_loss`, `val/repa_cos_sim` on the held-out set.

## 2. Config

Every knob lives under `[model.repa]` in the TOML (`RepaTomlConfig`, VFM only) and lands on
`model.config.repa.*` (`RepaConfig`):

| key | default | notes |
| --- | --- | --- |
| `enabled` | `true` (experiment) / `false` (framework) | |
| `relation_loss_weight` | `0.0` | weight of the VideoREPA-style token-relation loss `repa_rel_loss` (see below); v5 = `5.0` with `loss_weight = 0.0`. Always computed and logged, `0` = monitor only |
| `relation_distance` | `l2` | entry-wise distance between the two relation maps: `l2` (squared) or `l1` (absolute) |
| `loss_weight` | `0.5` | weight of `1 - cos`. NOTE: the flow-matching terms carry `loss_scale`/`action_loss_weight` = 10, so 0.5 is relatively ~20x weaker than the REPA paper's lambda=0.5 on an unscaled denoising loss; scan e.g. 0.5 / 2 / 5. |
| `objective` | `token` | `token` (base), `temporal_difference` (v8), or `spatial_normalized` (v10) |
| `spatial_norm_eps` | `1e-6` | denominator epsilon used by `spatial_normalized` |
| `layer_index` | `8` | blocks applied before the read (1-based). Nemotron-2B has 28. |
| `teacher` | `vjepa2_1_vit_base_384` | or `vjepa2_1_vit_large_384` (aliases `vitb` / `vitl`) |
| `teacher_checkpoint_path` | `None` | file or dir; default `$COSMOS_STORAGE/checkpoints/vjepa2_1/<release file>`; downloaded if absent |
| `teacher_input_size` | `256` | square side per view |
| `teacher_num_frames` | `16` | = `chunk_length` |
| `teacher_batch_size` | `32` | clips per teacher forward chunk (memory) |
| `target_adapter` | `avgpool` | `avgpool_conv`, `strided_conv` |
| `target_adapter_kernel_size` | `3` | variant 2 |
| `target_adapter_depthwise` | `true` | variant 3 |
| `target_grid_thw` | `[4, 5, 5]` | per-view token grid; needed to build variant 3, validated for all |
| `projector_type` | `mlp` | student-side projector: `mlp` = REPA `Linear-SiLU-Linear-SiLU-Linear`; `linear` = one `Linear(2048, D_t)` (v4 recipe: `h_k` itself has to become an affine image of the teacher features) |
| `projector_hidden_dim` | `2048` | MLP hidden width; ignored for `linear` |
| `num_views` | `2` | views concatenated along the canvas width |

The experiment adds `repa_` to `optimizer.keys_to_select` (trainable: `net.repa_head.projector.*`,
`net.repa_head.target_adapter.*`) and to `checkpoint.keys_to_skip_loading` (absent from the base DCP). The
teacher is held on `OmniMoTModel.repa_teacher` outside the `nn.Module` registry: not trained, not checkpointed,
not FSDP-wrapped. Context parallelism is not supported with REPA (raw frames are not part of the CP payload).

## 3. Run

```bash
cd ~/project/cosmos_hs10/packages/cosmos3
export LIBERO_ROOT=$COSMOS_STORAGE/data/LIBERO_LeRobot_v3/libero_10
export BASE_CHECKPOINT_PATH=$COSMOS_STORAGE/checkpoints/Cosmos3-Edge
export WAN_VAE_PATH=$COSMOS_STORAGE/checkpoints/wan22_vae/Wan2.2_VAE.pth
# teacher checkpoints: $COSMOS_STORAGE/checkpoints/vjepa2_1/vjepa2_1_vit{b,l}_dist_vitG_384.pt (already downloaded)
# optional: encoder-only files (faster start-up) -> python -m cosmos_framework.scripts.export_vjepa2_1_encoder --checkpoint <file> --verify

NPROC_PER_NODE=2 sr 2 48 bash examples/launch_sft_action_policy_libero_10_edge_repa.sh                        # ViT-B, block 8, variant 1
REPA_TOML_FILE=examples/toml/sft_config/action_policy_libero_10_edge_repa_l14.toml  NPROC_PER_NODE=2 sr 2 48 bash examples/launch_sft_action_policy_libero_10_edge_repa.sh
REPA_TOML_FILE=examples/toml/sft_config/action_policy_libero_10_edge_repa_vitl.toml NPROC_PER_NODE=2 sr 2 48 bash examples/launch_sft_action_policy_libero_10_edge_repa.sh
REPA_TOML_FILE=examples/toml/sft_config/action_policy_libero_10_edge_repa_v2.toml   NPROC_PER_NODE=2 sr 2 48 bash examples/launch_sft_action_policy_libero_10_edge_repa.sh
REPA_TOML_FILE=examples/toml/sft_config/action_policy_libero_10_edge_repa_v3.toml   NPROC_PER_NODE=2 sr 2 48 bash examples/launch_sft_action_policy_libero_10_edge_repa.sh
NPROC_PER_NODE=2 sr 2 48 bash examples/launch_sft_action_policy_libero_10_edge_repa_v4.sh                     # v4: linear projector (avgpool, block 8)
NPROC_PER_NODE=2 sr 2 48 bash examples/launch_sft_action_policy_libero_10_edge_repa_v5.sh                     # v5: token-relation L2 loss w5.0, direct cos off
NPROC_PER_NODE=2 sr 2 48 bash examples/launch_sft_action_policy_libero_10_edge_repa_v8.sh                     # v8: temporal-transition alignment
NPROC_PER_NODE=2 sr 2 48 bash examples/launch_sft_action_policy_libero_10_edge_repa_v9.sh                     # v9: base REPA + visual-token SIGReg
NPROC_PER_NODE=2 sr 2 48 bash examples/launch_sft_action_policy_libero_10_edge_repa_v10.sh                    # v10: per-frame spatial normalization
# any knob: EXTRA_TAIL_OVERRIDES="model.config.repa.loss_weight=2.0 job.name=hs10_repa_w2" ...
```

Smoke test (4 iters, 2 GPUs): `sr 2 48 bash ~/project/_scratch/hs10_repa_smoke.sh` (`VARIANT=v2|v3|v4|v5|vitl|l14`).
CPU tests: `PYTHONPATH=. python -m pytest --noconftest -c /dev/null -p no:cacheprovider --rootdir=. cosmos_framework/model/generator/repa cosmos_framework/configs/toml_config/repa_toml_test.py`.

## 4. Cost

Per optimizer step and rank (128 windows): the teacher encodes 256 clips of 2048 tokens; ViT-B ~0.5 TFLOP/clip
(a few seconds on an A6000, chunked 32 clips at a time), ViT-L ~3.5x that. The REPA head adds ~10 M params
(projector) + 0.6 M (variants 2/3). One extra native uint8 clip (6.7 MB) travels per sample through the
DataLoader; `train.py` therefore uses `file_system` tensor sharing and raises the fd soft limit.

## 5. Sanity expectations

* `repa_cos_sim` at iteration 0 (fresh projector) is ~0 and should rise quickly to 0.3-0.6 for variant 1; the
  flow-matching curves should track the hs08 baseline (`val/flow_matching_loss_action` is the comparison metric).
* The token grid printed in the error messages / asserted by `RepaAlignmentHead` must be `(5, 5, 10)` per sample;
  a different data pipeline (other resolution, single view) needs `target_grid_thw` / `num_views` adjusted.

## 6. DINOv2 teacher (v7)

`repa.teacher = "dinov2_vitb14"` swaps the frozen V-JEPA 2.1 video encoder for DINOv2 ViT-B/14 (the REPA paper's
teacher; `facebook/dinov2-base` via `transformers.Dinov2Model`, read from the local HF cache first). It is an image
model: every frame of every view is encoded separately at `teacher_input_size = 224` (16x16 patches of 14 px, the same
grid as V-JEPA at 256), giving `16 x 16 x 16` tokens per view; the unchanged `avgpool` adapter then pools 16 frames
into the 4 latent frames (4 consecutive frames each, exactly the Wan VAE mapping) and 16x16 -> 5x5 in space. Tokens are
DINOv2's post-LayerNorm patch tokens (`x_norm_patchtokens`, CLS dropped). Recipe: `action_policy_libero_10_edge_repa_v7.toml`
(= default recipe with the DINOv2 teacher). `dinov2_vits14` / `dinov2_vitl14` are available too. Pre-download once with
`python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/dinov2-base', allow_patterns=['*.json','*.safetensors'])"`.
