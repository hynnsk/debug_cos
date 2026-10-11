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
