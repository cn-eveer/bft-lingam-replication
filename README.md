# BFT LiNGAM Replication Package

This package computes every data-derived table and figure of the paper from the original bug-report data. No result is copied from the manuscript. The manuscript's values are used only in the last step, which compares them with the computed values.

## Inputs

| Path | Content |
|---|---|
| `data/raw/core/summary.csv`, `data/raw/product/summary.csv` | Original extracted Bugzilla@Mozilla reports (83,175 Core, 88,168 Product), the input of the original analysis code (`exec/lingam_s.py`, `exec/bn_t.py`) |
| `data/lingam/{core,product}/lingam_tce_user_not_newcomer_filter_wtf_fixed*.csv` | Original 100-sample bootstrap DirectLiNGAM outputs, 4 BFT classes (all reports + each class) |
| `data/lingam/{core,product}/lingam_tce_user_not_newcomer_filter_2_wtf_fixed*.csv` | Same, 2 BFT classes (Core ≤ 9 days, Product ≤ 4 days) |
| `data/models/{core,product}/bn_model_wtf*_hc_bdeu_none.pkl` | Original saved bnlearn BN structures from `exec/bn_t.py` (Feb/May 2025), used by the paper protocol because hill-climbing is not exactly deterministic |
| `data/paper_result/heatmap_{core,product}.csv` | Values plotted in the manuscript's Figures 5-6 (Core "Assigned Days(Class)" class 1 corrected to −0.063 on 2026-10-01); used only for the comparison step |

Bugzilla collection is not re-run.

## Install

```bash
pip install pandas numpy matplotlib seaborn scikit-learn networkx pgmpy bnlearn "lingam==1.12.2" jupyter
```

`lingam` 1.12 needs scikit-learn ≥ 1.6. Tested with Python 3.11 (pandas 2.1 and 3.0).

## Run

From this directory:

```bash
python analyze.py                      # all tables and figures, both protocols (~3 min)
python analyze.py --retrain-lingam     # also re-run the LiNGAM bootstrap from data/raw (~40 min)
python analyze.py --include-working-time   # sensitivity run, see "Predictors" below
```

Or open `01_data_collection.ipynb` and `02_analysis.ipynb`. `python generate_data.py` writes the preprocessed analysis datasets to `output/analysis_data/`.

Outputs go to `output/tables/` (CSV + Markdown) and `output/figures/`.

## Steps

| Step | Paper artifact | How it is computed |
|---|---|---|
| 01 | Data funnel | Row counts after each filter and input SHA-256 (`01_data_collection.csv`) |
| 02 | Table 1 | BFT class counts of the analysis subset (`lingam_s.process(filter=True)` + `check_train`: IQR on working time, severity/priority changed, FIXED, not self-reported, not newcomer) |
| 03 | Figures 5, 6 | Signed LiNGAM total effect of each factor on BFT, for all reports and each class (`lingam_s.evaluate_tce`), from `data/lingam/` or a fresh bootstrap |
| 04 | Table 3 | Paper rule applied to the Figure 5/6 values: S \|v\|>5, M 1-5, W <1, N none; Product/Core when they differ |
| 05 | Tables 4-7 | RF, LR, BN (hill-climbing, BDeu) and LiNGAM-BN (BN structure = LiNGAM edges with bootstrap probability ≥ 0.8, as in `bn_t.get_lingam_edge`) trained on a stratified 80/20 split, seed 42. Table 7: Cliff's delta vs RF over 10 stratified folds of the test set; paired permutation p-value (1,000 permutations) |
| 06 | Tables 8, 9 | Table 8: BFT of test reports that LiNGAM-BN predicts as class 1. Table 9: binary task with the 2-class LiNGAM outputs; LiNGAM+X use factors with a LiNGAM path to BFT with probability ≥ 0.8 |
| 07 | — | `07_paper_vs_computed.csv`: each manuscript value next to the computed one |
| 08 | Tables 4-7, 9 (paper protocol) | `paper_protocol.py`, a re-implementation of `exec/bn_t.py`, the code that produced the manuscript's BN / LiNGAM-BN results. Output in `output/paper_protocol/tables/`, including `08_protocol_diagnostics.csv` |

## Two evaluation protocols

`output/tables/` (**held-out**, default): models are trained on the 80 % split only, and `working_time` is not a predictor.

`output/paper_protocol/tables/` (**paper**): the protocol of `exec/bn_t.py`, which reproduces the manuscript. It differs in two ways that the manuscript does not state:

1. The BN and LiNGAM-BN are fitted on **all** reports, so the 20 % test reports are also in their training data.
2. `working_time` (creation to resolution, binned at the same 1/7/30-day cut points as BFT) is part of the evidence. `working_time` always contains `fixing_time`.

`08_protocol_diagnostics.csv` shows each condition's effect:

| Model | Fitted on | working_time | Core acc. | Product acc. |
|---|---|---|---|---|
| BN | all reports | yes | 0.779 | 0.789 |
| LiNGAM-BN | all reports | yes | **0.841** | **0.817** |
| BN | training split | yes | 0.773 | 0.789 |
| LiNGAM-BN | training split | yes | 0.706 | 0.720 |
| BN | all reports | no | 0.371 | 0.397 |
| LiNGAM-BN | all reports | no | 0.489 | 0.439 |

LiNGAM-BN is highest only when the test reports are part of its training data. Its 100-edge (Core) / 72-edge (Product) network fits those reports; on held-out reports BN is better.

## Predictors

`working_time` (creation to resolution) always contains `fixing_time`, so using it to predict the BFT class leaks the target. It is excluded from the prediction models by default. `--include-working-time` reproduces the leaky setting for comparison; the LiNGAM causal analysis uses all original columns in both cases.

## Reproducibility status

`python check_manuscript.py --section <path to the manuscript's section/ folder>` (run automatically by `analyze.py` when that folder is next to this package; the manuscript source is not part of this repository) reads every value printed in the manuscript's tables directly from the LaTeX source. It checks each against the computed value, to the precision the paper prints. Details are written to `output/manuscript_check.csv`.

| Manuscript table (current numbering) | Package output |
|---|---|
| Table I: BFT distribution | `table_1_*` |
| Fig. 5, 6: causal-strength heatmaps | `figure_5/6_*` |
| Table III: strength and direction | `table_3_*` |
| Table IV: P / R / F1 (Product + Core) | `paper_protocol/table_4_*`, `table_5_*` |
| Table V: macro-F1 | `paper_protocol/table_6_*` |
| Table VI: F1, Cliff's delta, p | `paper_protocol/table_7_*` |
| Table VII: short-duration BFT statistics | `paper_protocol/table_8_*` |
| Table VIII: binary accuracy | `paper_protocol/table_9_*` |

**Note on RF / LR.** No RF / LR code survives in `exec/`. The RF, LR, LiNGAM+LR and LiNGAM+RF results reported in the manuscript (revised 2026-10-01) are the reconstruction in this package (the BNs' inputs, fitted on the training split), as are the effect sizes, p-values and short-duration statistics, which the manuscript now reports as computed here.

The LiNGAM bootstrap is not bit-reproducible: `DirectLiNGAM.bootstrap` resamples without a fixed seed, so a retrained run differs slightly from `data/lingam/`. The figures and Table III therefore use the stored original outputs.
