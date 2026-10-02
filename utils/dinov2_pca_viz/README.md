# dinov2_pca_viz — LIBERO-10 frames through DINOv2 ViT-B/14, PCA-visualized next to the REPA target

What the frozen DINOv2 teacher of the cosmos_hs10 / cosmos_hs11 REPA loss "sees", per LIBERO-10 task:

```
row 1   16 raw frames (the predicted frames 1..16 of one 17-frame training window; third-person | wrist, 256 px each)
row 2   DINOv2 ViT-B/14 patch tokens of every frame (224 px input -> 16x16 patches per view) -> PCA(3) -> RGB
row 3   the distillation target: "avgpool" adapter = adaptive_avg_pool3d over (T,H,W) 16x16x16 -> 4x5x5 per view
        (= the 4x5x10 MoT token grid), projected with the SAME PCA basis / color stretch; one map per temporal bin,
        drawn under the frames it averages (4 frames -> 1 by default; --target-grid 8 5 5 gives 2 -> 1)
```

The window is built exactly like `LIBEROLeRobotDataset` (consecutive frames at the native 20 fps, frame 0 = clean
conditioning frame, not encoded) and the teacher / pooling steps mirror
`cosmos_hs11/.../model/generator/repa/{dinov2_teacher.py, adapters.py}`; the script itself is standalone (PyAV
decoding, no cosmos imports, no torchcodec).

## Run

```bash
bash utils/dinov2_pca_viz/run.sh                     # CPU: ~1-2 min for all 10 tasks (login node is fine)
sr 1 48 utils/dinov2_pca_viz/run.sh                  # GPU node: seconds
bash utils/dinov2_pca_viz/run.sh --help              # all options
```

Defaults: conda env `cosmos3-pt`, `LIBERO_ROOT=$COSMOS_STORAGE/data/LIBERO_LeRobot_v3/libero_10`,
`facebook/dinov2-base` from `$HF_HOME/hub` (offline when cached). Per task the FIRST episode of the 30-demo few-shot
training subset (`libero_10_3ep_per_task_seed42.json`) is used and the window starts at the middle of the episode.

Useful knobs (all passed through to `visualize_dinov2_pca.py`):

| flag | meaning |
|---|---|
| `--tasks 0,3,7` | subset of task indices (default all 10) |
| `--window-pos 0.25` / `random` / `--start-frame N` | where the window starts in the episode |
| `--episode-pick random --seed 1` / `--episode-subset none` | another episode of the task |
| `--target-grid T H W` | the pooled grid per view (default `4 5 5`; T=8 -> 2 frames per map) |
| `--pca-per-view` | separate PCA per camera view (the joint fit spends PC1 on 3rd-person vs wrist) |
| `--pooled-pca refit` | row 3 gets its own PCA instead of the shared basis (more contrast, colors not comparable) |
| `--pca-scope global` | one PCA basis for all tasks (colors comparable across PNGs) |
| `--teacher dinov2_vitl14`, `--input-size 252` | other DINOv2 sizes / grids |
| `--save-npz` | also dump tokens `[V,T,16,16,768]`, pooled targets `[V,4,5,5,768]`, frames (float16 / uint8) |

## Output

```
utils/dinov2_pca_viz/results/<run-name>/          run-name default: dinov2_vitb14_in224_pool4x5x5
    task00_turn_on_the_stove_and_put_the_moka_pot_on_it.png
    ...
    task09_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.png
    run_info.json                                 episode / window / PCA explained variance per task + the args
```

Reading the colors: PC1 -> R, PC2 -> G, PC3 -> B after a 1-99 percentile stretch fitted on the window's full-resolution
tokens. With the default shared basis the pooled maps are literally the (clipped) average of the row-2 colors over
each 4x(adaptive 16->5) bin, so paler / blurrier colors in row 3 are the information the avgpool target throws away.
