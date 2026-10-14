# Meta Pre-Training World Action Models for Few-Shot Robot Post-Training (CVPR 2026 draft)

LaTeX project built on the official CVPR 2026 author kit (`cvpr.sty`, `ieeenat_fullname.bst`, review mode).

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`algorithm.sty`, `algorithmicx.sty`, `algpseudocode.sty` are bundled only because the local TeX Live lacks them;
Overleaf has them and the local copies are harmless there. Switch `\usepackage[review]{cvpr}` to `\usepackage{cvpr}`
for the camera-ready version.

## Layout

| Path | Content |
| --- | --- |
| `main.tex` | title, authors, section includes |
| `preamble.tex` | packages, macros (`\base`, `\basept`, `\metapt`, `\thmeta`, `\method`) |
| `sec/0_abstract.tex` … `sec/5_conclusion.tex` | abstract, introduction, related work, method, experiments, conclusion |
| `fig/method.tex` | Figure 2 (TikZ overview of the pipeline and one Reptile meta-iteration) |
| `fig/fig1_data_scarcity.{pdf,png}` | Figure 1 (all demos vs. 3 demos/task); regenerate with `fig1.py` (matplotlib, Nimbus Roman) |
| `main.bib` | references |

## Where the numbers come from

* Table 1, LIBERO-10 column, and Table 2 (per task): `cosmos_storage/outputs/eval/cosmos3_action_libero/edge_libero10_fewshot/hs08_v2_50demo_2000`
  (PT, 58.0) and `.../edge_libero10_3ep_fullft_reptile_v4_init_50demo_2000` (META-PT, 69.4), both evaluated with guidance 1.0.
* Table 3 (control): JOINT-PT 58.2 = `.../edge_libero10_3ep_fullft_sourceinit_50demo_2000`, post-trained from
  `cosmos3_action_meta/reptile_meta_ablation/edge_source_sft_joint_k16w16_bs256_1k_seed42` (k=1, eps=1, 1000 episodes of
  16 demos x 16 windows).
* Table 1 Goal/Spatial/Object columns (63.2/76.2, 74.8/78.6, 90.6/90.4), the "all demonstrations" row and Table 4
  (87.2 / 94.6) are the numbers given in the writing brief (2026-10-14); they are not in `cosmos_storage/outputs/eval` on this
  machine. The paper states that META-PT uses ranking policy learning (RPL, the v23 `[model.language_ranking]` objective)
  on Goal and Spatial only; confirm which arms used RPL and whether their evaluation used classifier-free guidance.
* Post-training hyper-parameters: `config.yaml` of the two LIBERO-10 runs above. Meta pre-training hyper-parameters:
  `cosmos3_action_meta/reptile_meta/edge_reptile_v4_my/config.yaml` (K=32, w=8, k=10, eps 0.5->0.05, 1000 episodes).
  RPL hyper-parameters: `examples/toml/sft_config/action_policy_libero_goal_edge_reptileinit_v23.toml` and
  `docs/action_policy_libero_language_ranking.md` in `cosmos_hs11`.
