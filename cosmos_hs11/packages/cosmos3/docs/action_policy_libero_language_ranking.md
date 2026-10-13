# LIBERO action-language ranking after Reptile initialization

Four recipes add action-language ranking to the existing WAM post-training objective. They use no
distillation teacher and add no model parameters. All optimizer, schedule, batch, EMA and checkpoint
settings are copied from the specified source files, including local edits present when copied.

| Suite | New launcher / TOML suffix | Source launcher | Actual source TOML |
| --- | --- | --- | --- |
| LIBERO-10 | `libero_10_edge_reptileinit_v58` | `libero_10_edge_reptileinit_v3.sh` | `libero_10_edge_reptileinit_v3.toml` |
| LIBERO-Goal | `libero_goal_edge_reptileinit_v23` | `libero_goal_edge_reptileinit_v13.sh` | `libero_goal_edge_reptileinit_v13.toml` |
| LIBERO-Object | `libero_object_edge_reptileinit_v23` | `libero_object_edge_reptileinit_v3.sh` | `libero_object_edge_reptileinit_v3.toml` |
| LIBERO-Spatial | `libero_spatial_edge_reptileinit_v23` | `libero_object_edge_reptileinit_v7.sh` | `libero_object_edge_reptileinit_v7.1.toml` |

Launchers have prefix `examples/launch_sft_action_policy_`; TOMLs have prefix
`examples/toml/sft_config/action_policy_`. Spatial inherits Object v7.1's hyperparameters and uses
the Spatial train/validation subset JSONs. Goal retains its source's `libero_goal_3ep_per_task_v2.json`.

## Objective and sampling

For each window, the LIBERO dataset reads its task ID and uniformly samples one of the other **nine
task IDs** from that dataset's task metadata. No in-batch negatives or cross-suite instructions are
used. If a task has caption alternatives separated by ` | `, one alternative is sampled after the
task is chosen. Exactly ten nonempty task entries are required; use a separate `LIBERO_ROOT` per suite.
Sampling uses the existing seeded Python/DataLoader worker RNG.

The negative JSON prompt changes only `actions[0].description`. Camera, time, resolution and all
other prompt metadata stay the same. The original positive prompt and its CFG dropout are preserved.
Samples whose positive prompt is dropped contribute zero ranking loss; the ordinary FM loss still
trains on them. Validation has the source recipe's zero CFG dropout and also reports ranking metrics.

```
E_pos[i] = mean((v_theta(xt, t, observation, instruction_pos) - v_target)^2)
E_neg[i] = mean((v_theta(xt, t, observation, instruction_neg) - v_target)^2)
L_rank   = mean_i(valid_instruction[i] * relu(margin + E_pos[i] - E_neg[i]))
L_total  = L_original + loss_weight * L_rank
```

Each action error is calculated in fp32 over **noisy action timesteps and real, valid action
channels only**. Conditioned actions, invalid unified slots and padding are excluded. The errors
do not include the original action FM multiplier or diffusion time weighting. The hinge is applied
per sample before the batch mean. Dropped samples contribute zero to the original-batch mean,
avoiding per-rank normalization differences when CFG dropout counts differ.

Positive and negative branches share the exact noised video/action tensors, clean conditioning,
flow-matching target and timesteps. Only language changes. Both branches receive gradients. They
execute inside one root network forward for FSDP; the negative branch decodes actions only and adds
no negative-video reconstruction objective. Expect an additional transformer forward/backward and
higher activation memory; the copied batch sizes have not been reduced automatically.

The requested baseline samples negatives from all other tasks, including task pairs whose local
action chunks may overlap. There is no task-pair filtering or special temporal weighting.

## Configuration and launch

Each new TOML includes:

```toml
[model.language_ranking]
enabled = true
loss_weight = 1.0
margin = 0.1
```

These are starting hyperparameters, not tuned results. Ranking is disabled by default for existing
recipes. Set `enabled = false` to remove both the negative sampling and extra forward.

Example, from `packages/cosmos3`, with checkpoint environment variables already set:

```bash
LIBERO_ROOT=/path/to/libero_10 NPROC_PER_NODE=2 bash examples/launch_sft_action_policy_libero_10_edge_reptileinit_v58.sh
LIBERO_ROOT=/path/to/libero_goal NPROC_PER_NODE=2 bash examples/launch_sft_action_policy_libero_goal_edge_reptileinit_v23.sh
LIBERO_ROOT=/path/to/libero_object NPROC_PER_NODE=2 bash examples/launch_sft_action_policy_libero_object_edge_reptileinit_v23.sh
LIBERO_ROOT=/path/to/libero_spatial NPROC_PER_NODE=2 bash examples/launch_sft_action_policy_libero_spatial_edge_reptileinit_v23.sh
```

Required checkpoint variables remain `REPTILE_CKPT_PATH`, `META_ACTION_INIT_PATH`, and `WAN_VAE_PATH`
(or the launcher's usual VAE default). `MASTER_PORT` remains overridable; the four defaults are distinct.

Training and validation log `language_ranking_loss`, `language_ranking_weighted_loss`,
`language_ranking_positive_error`, `language_ranking_negative_error`, `language_ranking_gap`
(`E_neg - E_pos`), `language_ranking_margin_satisfied`, and `language_ranking_valid_fraction`.
Error/gap/satisfaction diagnostics are local valid-sample means subsequently averaged by the existing
logging callbacks; the optimized loss itself uses the original-batch mean.

## Implementation

- Negative selection: `data/generator/action/utils/language_ranking.py` and the LIBERO dataset.
- Prompt preparation: `data/generator/action/utils/transforms.py`.
- CP-aligned metadata and identical-noise packing: `model/generator/omni_mot_model.py`.
- Paired network forwards: `model/generator/mot/cosmos3_vfm_network.py`.
- Masked hinge loss: `model/generator/algorithm/loss/language_ranking.py`.

Paths above are relative to `cosmos_framework/`. Tests cover sampling, paired prompts, collation,
gradient direction/masking, loss assembly, network checkpoint recomputation, shared noising and
recipe inheritance. CPU tests do not substitute for a full-checkpoint multi-GPU FSDP smoke run.

Verification for this change: 35 targeted CPU tests passed; all four recipes composed through
`load_experiment_from_toml`; launcher syntax, Ruff checks on changed Python files and `git diff --check`
passed. CUDA was unavailable, so no full-checkpoint GPU training was launched.

A separately run existing test, `libero_sampling_toml_test.py::test_v43_routes_the_knob_and_v3_does_not`,
fails because it assumes the v43 and v3 optimizer settings are equal. Their committed source values
already differ (`action2llm` multiplier 1.0 versus 2.0); these original recipes were preserved.

## v27: action-similarity hard negatives (2026-10-11)

Closed-loop GIFs of the goal / spatial 3-demo policies show that failures are *task confusion*: the policy executes
another task's behaviour (plate push -> bowl lift, bottle -> stove knob) or picks the wrong bowl. The uniform negative
of v23 spends most of its margin on pairs the policy never confuses. v27 keeps the loss and draws the negative
instruction's task from the other nine with a distribution built from the **training actions themselves**:

```
prefix_t   = first hard_negative_prefix_frames training actions (rot6d, un-normalized) of every KEPT demo of task t
mean_t     = per-time-step mean over task t's demos (channels standardized over all demos; shorter demos stop early)
d[i, j]    = RMS(mean_i - mean_j) over the time steps both cover and all channels
z          = d / median_{i != j}(d)                                  (scale-free; z = 0 on the diagonal)
p_hard(j|i)= softmax_{j != i}(-z[i, j] / hard_negative_temperature)
p(j|i)     = (1 - hard_negative_fraction) / 9 + hard_negative_fraction * p_hard(j|i)
```

Tasks whose demonstrations START alike (the window the policy decides in while vision cannot tell the tasks apart)
are the ones that get sampled, and the uniform floor keeps every pair in play. The weights are computed once per
dataset at construction from the kept subset (train json for the train loader, val json for the val loader), logged
as a matrix + top-3 table, and the per-window draw is `random.choices` over the sampler's candidate order. With
`negative_sampling = "uniform"` (or `hard_negative_fraction = 0`) the sampler is the v23 one, bit for bit.

| TOML key (`[model.language_ranking]`) | v27 | meaning |
| --- | --- | --- |
| `negative_sampling` | `"action_similarity"` | `"uniform"` = v23; the three knobs below are then ignored |
| `hard_negative_fraction` | 0.7 | mixture weight of the similarity distribution (0 = uniform, 1 = similarity only) |
| `hard_negative_temperature` | 0.5 | softmax temperature on `z` (median off-diagonal = 1); lower = sharper |
| `hard_negative_prefix_frames` | 64 | leading frames per demo (3.2 s at 20 fps); clipped to the demo length |

Recipes (three arms per suite; one knob differs per arm so the comparison stays interpretable; the uniform sibling
is the zero point of the hardness axis). LIBERO-10 TOMLs are the v3 recipe verbatim + the section, run names differ:

| arm | LIBERO-goal (v23 sibling) | LIBERO-10 (v58 sibling) | fraction / temperature | loss_weight | ports |
| --- | --- | --- | --- | --- | --- |
| hard negatives | `goal_edge_reptileinit_v27` | `10_edge_reptileinit_v63` | 0.7 / 0.5 | 1.0 | 50127 / 50130 |
| strength (x5) | `goal_edge_reptileinit_v28` | `10_edge_reptileinit_v64` | 0.7 / 0.5 | 5.0 | 50128 / 50131 |
| sharpness | `goal_edge_reptileinit_v29` | `10_edge_reptileinit_v65` | 1.0 / 0.25 | 1.0 | 50129 / 50132 |

TOML prefix `examples/toml/sft_config/action_policy_libero_`, launcher prefix `examples/launch_sft_action_policy_libero_`;
`margin` 0.1 and `hard_negative_prefix_frames` 64 everywhere. Why x5 and not x10: at 1.0 the ranking gradient is
1/10 of the action-FM gradient (same MSE scale, 10x FM multiplier) and the hinge is bounded by the margin, so a null
result would not separate 'wrong signal' from 'too weak'; at 10 the bounded hinge x 10 rivals the whole total loss and
steals clip budget from the FM terms. Watch `language_ranking_margin_satisfied`: ~0 means the absolute margin is
unattainable on near-identical prefixes (lower it, e.g. 0.05), ~1 early means the hinge is inactive. The knobs route
`[model.language_ranking].* -> model.config.language_ranking.* -> dataloader_{train,val}...libero.dataset.language_ranking_*`
(`LanguageRankingConfig`, `LanguageRankingTomlConfig`, `action_policy_libero_edge._libero_packing_loader`,
`get_action_libero_sft_dataset`, `LIBEROLeRobotDataset._build_instruction_negatives`). Other suites: copy the four
keys into their v23 TOML (the weights are computed from whichever subset the TOML names).

Inspect the distribution a recipe will train with (CPU, parquet only, a few seconds):

```bash
PYTHONPATH=. python cosmos_framework/scripts/print_libero_hard_negatives.py \
    --root $COSMOS_STORAGE/data/LIBERO_LeRobot_v3/libero_goal --episode-subset libero_goal_3ep_per_task_v2.json \
    --fraction 0.7 --temperature 0.5 --prefix-frames 64
```

Implementation: `data/generator/action/utils/language_ranking.py` (`action_prefix_distance_matrix`,
`hard_negative_weights`, `format_hard_negative_report`, weighted `LiberoInstructionNegatives`), the LIBERO dataset
and its factory, the experiment config, `model_config.py`, `sft_config.py`. Tests:
`utils/language_ranking_test.py` (weights + matrix math), `datasets/libero_hard_negatives_test.py` (synthetic
10-task LeRobot layout through the real constructor, subset handling, validation),
`configs/toml_config/language_ranking_toml_test.py` (v27 inheritance, routing, pydantic bounds, loader wiring).
Not GPU-smoked.

## v24 family: asymmetric gradient, relative margin, self-paced hinge (2026-10-13)

Closed-loop results of the uniform-negative recipes (50 trials/task, one seed): goal and spatial gain on top of the
Reptile init, LIBERO-10 drops (69.6 -> 64.8) and object drops (85 -> 82) with the dominant object failure being
"grasps another object". Two properties of the absolute hinge are suspected: the model can satisfy it by making the
wrong-instruction prediction worse (noise through the text pathway into the action head), and windows whose motion does
not depend on the instruction (lift / transport / release: 87 % of object chunks have a near-identical chunk in another
task) can never reach an absolute gap, so the hinge never switches off there. Three optional, loss-side modifications
address this; negatives stay uniform and the forward pass is unchanged
(`model/generator/algorithm/loss/language_ranking.py`):

```
E_neg_g   = s * E_neg + (1 - s) * sg(E_neg)                      # negative_grad_scale s: 1 = original, 0 = positive-only
margin_i  = margin                      (margin_mode = "absolute")
          = margin * sg(E_pos_i)        (margin_mode = "relative":  E_neg >= (1 + margin) * E_pos)
hinge_i   = relu(margin_i - (E_neg_g_i - E_pos_i))
L_rank    = sum_i(hinge_i * valid_i * sel_i) / (B * q)          # self_paced_fraction q: sel = easiest ceil(q * n_valid)
                                                                #   valid samples by detached gap_i - margin_i; q = 0 -> sel = valid, / B
```

| TOML key (`[model.language_ranking]`) | default | meaning |
| --- | --- | --- |
| `negative_grad_scale` | 1.0 | gradient scale through `E_neg`; the loss value is unchanged. 0.0 = positive-only (the hinge can only lower `E_pos`). |
| `margin_mode` | `"absolute"` | `"relative"` scales `margin` by the sample's own detached `E_pos`. |
| `self_paced_fraction` | 0.0 | 0 = off; otherwise only this fraction of the valid samples with the largest detached `gap - margin_i` is ranked, normalized by `batch * fraction`. |

New metrics: `language_ranking_effective_margin` (mean `margin_i` over valid samples) and
`language_ranking_selected_fraction` (selected / valid; 1.0 when self-pacing is off). `margin_satisfied` uses `margin_i`.

Recipes: the uniform sibling verbatim (object v23 / LIBERO-10 v58) + run name + the knobs. All four arms use
`negative_grad_scale = 0.2` (asymmetric; set 0.0 for positive-only), `loss_weight = 1.0`:

| arm | object | LIBERO-10 | margin_mode / margin | self_paced_fraction | ports |
| --- | --- | --- | --- | --- | --- |
| asymmetric gradient only | `object_edge_reptileinit_v24` | `10_edge_reptileinit_v75` | absolute / 0.1 | 0 | 50224 / 50175 |
| + relative margin | `object_edge_reptileinit_v25` | `10_edge_reptileinit_v76` | relative / 0.3 | 0 | 50225 / 50176 |
| + self-paced hinge | `object_edge_reptileinit_v26` | `10_edge_reptileinit_v77` | absolute / 0.1 | 0.5 | 50226 / 50177 |
| + both | `object_edge_reptileinit_v27` | `10_edge_reptileinit_v78` | relative / 0.3 | 0.5 | 50227 / 50178 |

Relative margin 0.3 matches the absolute 0.1 at the Reptile init's val action error (~0.33) and then shrinks with it.
Self-paced 0.5 keeps every batch's easier half; watch `language_ranking_selected_fraction` (0.5 by construction) together
with `margin_satisfied` and `gap`: if the selected half saturates while `gap` of the rest stays ~0, the excluded windows
are the instruction-independent ones. The knobs route like the others
(`LanguageRankingTomlConfig` -> `model.config.language_ranking.*` -> `LanguageRankingConfig` -> `_compute_losses`).
Tests: `loss/language_ranking_test.py` (value/gradient of each knob, production assembly) and
`configs/toml_config/language_ranking_toml_test.py` (sibling + knobs structure, routing, launchers, ports). Not GPU-smoked.
