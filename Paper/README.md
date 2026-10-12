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

* Table 1, LIBERO-10 column: `cosmos_storage/outputs/eval/cosmos3_action_libero/edge_libero10_fewshot/hs08_v2_50demo_2000`
  (58.0) and `.../edge_libero10_3ep_fullft_reptile_v4_init_50demo_2000` (69.4). Table 2 (per task) is read from the same
  two `summary.json` files.
* Table 1, Goal/Spatial/Object columns, the "all demonstrations" row, and Table 3 (87.2 / 94.6) are the numbers given in
  the writing brief; they are not in `cosmos_storage/outputs/eval` on this machine and should be re-checked against the
  runs that produced them before submission.
* Post-training hyper-parameters: `config.yaml` of the two LIBERO-10 runs above (identical except for the init and the
  action-head LR multiplier). Reptile hyper-parameters: `examples/toml/sft_config/action_reptile_meta_edge_v4.toml` and
  the `edge_reptile_v4*` run logs in `cosmos_hs11`.
