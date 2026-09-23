# Meta init + DINOv2 REPA distillation for LIBERO-10 post-training (cosmos_hs09_2)

`cosmos_hs09_2` = a copy of `cosmos_hs09` (LoRA-regime few-shot meta-training + meta-initialized LIBERO-10
post-training, see `docs/action_fewshot_meta_lora.md`) **plus** the `cosmos_hs10` representation-alignment (REPA)
loss (see `docs/action_policy_libero_repa_vjepa.md`), so that the meta-initialized post-training can additionally
distill frozen DINOv2 ViT-B/14 features into the MoT generation pathway (= the hs10 `v7` setting).

```
init        : theta_meta (LoRA r32 adapters + action heads + time_embedder) from action_fewshot_meta_lora_edge,
              copied into net / net_ema right after the base-DCP load  (cosmos_hs09, unchanged)
trainable   : full moe_gen + time_embedder + vae2llm/llm2vae + k_norm_und_for_gen + action heads   (hs08 recipe)
              + net.repa_head.*  (REPA MLP 2048->2048->2048->768; the avgpool adapter has no parameters)
              LoRA carrier frozen via lr multiplier 0 (== the meta-learned delta merged into moe_gen)
loss_total  = 10 * fm_vision + 10 * fm_action                                                    (unchanged)
            + 5.0 * ( 1 - cos( MLP(h_8[predicted video tokens]), avgpool(DINOv2-ViT-B/14(frames 1..16)) ) )
```

`h_8` is the residual stream after MoT block 8 (of 28) of the *predicted* (noised) video tokens (latent frames
1..4, 5x10 tokens each = 4x5x10 per window; latent frame 0 is the clean conditioning frame and is never
aligned). The teacher is `facebook/dinov2-base` (frozen, not checkpointed, not sharded), applied per frame to the
16 future frames of **each camera view** at 224 px (16x16 patches of 14 px, post-LayerNorm patch tokens); the
parameter-free `avgpool` adapter pools 16 frames -> 4 latent frames (bins of 4 = the Wan VAE mapping) and 16x16 ->
5x5 per view, then the two views are concatenated along the width like the `concat_view` canvas.

## Files (what changed w.r.t. cosmos_hs09)

| Piece | Path |
| --- | --- |
| Recipe TOML (**the** deliverable) | `examples/toml/sft_config/action_policy_libero_10_edge_metainit_repa_dinov2.toml` = `action_policy_libero_10_edge_metainit.toml` + `[model.repa]` (hs10 v7) + `"repa_"` in `keys_to_skip_loading` + experiment/name |
| Launcher | `examples/launch_sft_action_policy_libero_10_edge_metainit_repa_dinov2.sh` (port 50018, `REPA_TOML_FILE` override) |
| Experiment | `configs/base/experiment/action/posttrain_config/action_policy_libero_edge_repa.py` = hs10's experiment + the hs09 `lora_` LR-multiplier hook; registered in `configs/base/config.py` |
| REPA code (ported 1:1 from cosmos_hs10) | `model/generator/repa/` (adapters, alignment head, DINOv2 + V-JEPA teachers, teacher registry, SIGReg), capture in `mot/unified_mot.py::_impl_forward`, `repa_pred/repa_target` in `mot/cosmos3_vfm_network.py`, teacher call + loss term in `omni_mot_model.py` (`_set_up_repa_teacher`, `_compute_repa_teacher_tokens`, `_compute_losses`) |
| Config schema | `configs/base/defaults/model_config.py::RepaConfig` (+ `SigRegConfig`), `configs/toml_config/sft_config.py::RepaTomlConfig` (`[model.repa]`), `toml_config_helper.py` VLM blocklist |
| Data | `keep_native_video=True` in the REPA experiment's loaders -> `data_batch["video_native"]` (uint8 3x17x256x512) for the teacher (`action_sft_dataset.py`, `transforms.py`, `joint_dataloader.py`) |
| Robustness fixes that came with hs10 | `scripts/train.py` (file_system tensor sharing + RLIMIT_NOFILE), `utils/distributed.py::warm_up_checkpoint_collectives` (+ call in `utils/config.py`), `torch.cuda.empty_cache()` before `dcp.save`, `wandb_log.py` logs `repa_cos*` |
| Tests | `model/generator/repa/*_test.py`, `configs/toml_config/repa_toml_test.py` (now also checks the meta-init lines of the new TOML), `utils/distributed_test.py` |

Everything the meta init touches is byte-identical to `action_policy_libero_10_edge_metainit.toml` (LoRA carrier
config, `lora_freeze_base=false`, lr / multipliers / schedule / batch / validation, `[checkpoint].meta_action_init_*`),
so the pair (metainit, metainit + REPA) isolates the distillation loss. The meta init never touches `repa_head`
(theta_meta has no such group; `apply_meta_action_init_to_net` only writes the groups the file carries), and
`repa_head` is skipped at the base-DCP load because the base has no such tensors.

## Commands

```bash
cd ~/project/cosmos_hs09_2/packages/cosmos3
export LIBERO_ROOT=$COSMOS_STORAGE/data/LIBERO_LeRobot_v3/libero_10
export BASE_CHECKPOINT_PATH=$COSMOS_STORAGE/checkpoints/Cosmos3-Edge
export WAN_VAE_PATH=$COSMOS_STORAGE/checkpoints/wan22_vae/Wan2.2_VAE.pth
# theta_meta from the cosmos_hs09 meta run (an iteration snapshot or meta_action_init.pt = latest)
export META_ACTION_INIT_PATH=$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action_meta/fewshot_meta_lora/edge_meta_lora_r32_fomaml_adam_k5q5_w8_seed42/meta_action_init_iter_001000.pt

# (0) config only
PYTHONPATH=. python -m cosmos_framework.scripts.train --sft-toml=examples/toml/sft_config/action_policy_libero_10_edge_metainit_repa_dinov2.toml --dryrun

# (1) meta init + DINOv2 REPA, 2 GPUs (shard 2, 128 windows/rank -> global 256, 2000 iters, val at 0 and every 200)
NPROC_PER_NODE=2 sr 2 48 examples/launch_sft_action_policy_libero_10_edge_metainit_repa_dinov2.sh

# (2) the matching control without the distillation loss (unchanged cosmos_hs09 recipe)
NPROC_PER_NODE=2 sr 2 48 examples/launch_sft_action_policy_libero_10_edge_metainit.sh

# knobs without editing the TOML, e.g. weight / layer / V-JEPA teacher instead of DINOv2
EXTRA_TAIL_OVERRIDES="model.config.repa.loss_weight=1.0 model.config.repa.layer_index=14" NPROC_PER_NODE=2 sr 2 48 examples/launch_sft_action_policy_libero_10_edge_metainit_repa_dinov2.sh
EXTRA_TAIL_OVERRIDES="model.config.repa.teacher=vjepa2_1_vit_base_384 model.config.repa.teacher_input_size=256" ...   # needs $COSMOS_STORAGE/checkpoints/vjepa2_1/
```

The teacher weights are read from the Hugging Face cache (`$HF_HOME/hub/models--facebook--dinov2-base`) first and
downloaded once if absent; compute nodes have no internet, so make sure the snapshot exists (it does on this
cluster, see `cosmos_hs10`). CPU tests: `PYTHONPATH=. python -m pytest --noconftest -c /dev/null -p no:cacheprovider
--rootdir=. cosmos_framework/model/generator/repa cosmos_framework/configs/toml_config/repa_toml_test.py`.

## What to look at in wandb

* `val/flow_matching_loss_action` (primary), `val/flow_matching_loss_vision`, `val/loss_total` -- against the
  cosmos_hs09 `edge_libero10_3ep_fullft_metainit_*` run and the cosmos_hs08 fresh-init baseline.
* `repa_loss` / `val/repa_loss` (= 1 - cos, weight 5.0 in the total), `repa_cos_sim` (raw cosine; saturates ~0.9+
  quickly because DINOv2/V-JEPA tokens share a dominant mean direction), `repa_cos_sim_centered` (the informative
  one: cosine after removing the batch-mean teacher direction). `repa_rel_loss` is logged with weight 0 (monitor only).
* Iteration 0 (`run_validation_on_start`) is the meta-initialized model before any update; the REPA head is fresh
  there, so `val/repa_loss` ~ 1.0 at iteration 0 is expected.

## Caveats

* Memory: the hs10 REPA runs at 128 windows/rank x 2 ranks sit at the 44 GiB edge (A40/L40S); prefer the 48 GiB
  A6000 nodes or lower `max_samples_per_batch` (and raise `grad_accum_iter`) if it OOMs. The LoRA carrier adds ~33M
  frozen params on top of the hs10 footprint.
* `loss_weight=5.0` is the hs10 default for this teacher; relative to REPA's lambda it is weaker than it looks because
  the flow-matching terms are x10-scaled (`docs/action_policy_libero_repa_vjepa.md`).
* The whole hs10 REPA module was ported (V-JEPA 2.1 teacher, relation loss, centered targets, SIGReg, ...) so every
  hs10 variant is reachable through `[model.repa]` / `[model.sigreg]` or `EXTRA_TAIL_OVERRIDES`; only the DINOv2 v7
  recipe ships as a TOML here.
