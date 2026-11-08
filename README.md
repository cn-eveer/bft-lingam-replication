# BFT LiNGAM Replication Package

This package computes every data-derived table and figure of the paper from the original bug-report data. No result is copied from the manuscript. The manuscript's values are used only in the last step, which compares them with the computed values.

## Inputs

| Path | Content |
|---|---|
| `data/raw/core/summary.csv`, `data/raw/product/summary.csv` | Original extracted Bugzilla@Mozilla reports (83,175 Core, 88,168 Product), the input of the original analysis code (`exec/lingam_s.py`, `exec/bn_t.py`) |
| `data/lingam/{core,product}/lingam_tce_user_not_newcomer_filter_wtf_fixed*.csv` | Original 100-sample bootstrap DirectLiNGAM outputs, 4 BFT classes (all reports + each class) |
| `data/lingam/{core,product}/lingam_tce_user_not_newcomer_filter_2_wtf_fixed*.csv` | Same, 2 BFT classes (Core ≤ 9 days, Product ≤ 4 days) |
| `data/models/{core,product}/bn_model_wtf*_hc_bdeu_none.pkl` | Saved bnlearn BN structures used by the paper protocol, because hill-climbing is not exactly deterministic: the original models from `exec/bn_t.py` (Feb/May 2025), except the Core 2-class BN, whose original file had been overwritten and which was re-learned once with the same procedure |
| `data/lingam_train/lingam_tce_train_{core,product}_{iqr,no_iqr}.csv` | DirectLiNGAM bootstrap (100 samples) learned on the **training split only**, with BFT as the continuous fixing time; the LiNGAM-BN structure of the prediction steps 05-07 (`iqr`: outliers removed as in the paper, `no_iqr`: sensitivity analysis) |
| `data/lingam_refined/*.csv` | Direct-edge DirectLiNGAM bootstraps (100 samples) without total resolution time and with temporal prior knowledge, the structure of the refined LiNGAM-BN: `train_*` (training splits), `all_*` and `temporal_*` (transfer training sets); `grid/` holds the extra variants scored by `refine_lingam_bn.py` |
| `data/bn_structures/*.csv` | Hill-climbing structures of the standard BN in steps 05-07 (one per scope, task and outlier setting), learned once and reused: pgmpy breaks score ties in an order that depends on the Python process, so re-learning can give slightly different structures. `--relearn-bn` learns them again |
| `data/paper_result/heatmap_{core,product}.csv` | Values plotted in the manuscript's Figures 5-6 (Core "Assigned Days(Class)" class 1 corrected to −0.063 on 2026-10-01); used only for the comparison step |

Bugzilla collection is not re-run.

## Install

```bash
pip install pandas numpy matplotlib seaborn scikit-learn networkx "pgmpy==0.1.25" "bnlearn==0.10.2" "lingam==1.12.2" jupyter
```

`lingam` 1.12 needs scikit-learn ≥ 1.6. pgmpy and bnlearn must be pinned: bnlearn ≥ 1.0 changed `bnlearn.load`, and step 08 then fails (steps 01-07 and 09 give identical tables with pgmpy 1.1 / bnlearn 1.1). Tested with Python 3.11, numpy 2.4, pandas 2.1, 3.0 and 3.1.

## Run

From this directory:

```bash
python analyze.py                      # all tables and figures (~4 min)
python analyze.py --retrain-lingam     # also re-run every LiNGAM bootstrap from data/raw (~70 min)
python analyze.py --relearn-bn         # also re-learn the standard BN structures
python analyze.py --include-working-time   # sensitivity run, see "Predictors" below
```

Or open `01_data_collection.ipynb` and `02_analysis.ipynb`. `python generate_data.py` writes the preprocessed analysis datasets to `output/analysis_data/`.

Outputs go to `output/tables/` (CSV + Markdown) and `output/figures/`.

## Steps

| Step | Manuscript artifact | How it is computed |
|---|---|---|
| 01 | Fig. 2, Data section | Row counts after each filter and input SHA-256 (`01_data_collection.csv`); extracted and FIXED reports per project (`figure_2_data_overview.csv`: 171,343 extracted, 50,260 FIXED) |
| 02 | Table I | BFT class counts of the analysis subset (`lingam_s.process(filter=True)` + `check_train`: IQR on total resolution time, severity/priority changed, FIXED, not self-reported, not newcomer) |
| 03 | Figs. 5, 6 | Signed LiNGAM total effect of each factor on BFT, for all reports and each class (`lingam_s.evaluate_tce`), from `data/lingam/` or a fresh bootstrap |
| 04 | Table III | Rule applied to the Figure 5/6 values: S \|v\|>5, M 1-5, W <1, N none; Product/Core when they differ |
| 05 | Tables IV-VI | RF, LR, BN (hill-climbing, BDeu), LiNGAM-BN (BN structure = training-split LiNGAM total effects with bootstrap probability ≥ 0.8) and the refined LiNGAM-BN (direct LiNGAM edges with temporal prior knowledge, probability ≥ 0.8, at most 3 parents) on the 4 BFT classes. Stratified 80/20 split, seed 42. Table VI: Cliff's delta vs RF over 10 stratified folds of the test set; paired permutation p-value (1,000 permutations) |
| 06 | Table VII | Binary task, short ≤ 7 days (median BFT of both scopes) vs long, on the same split. LiNGAM+X use only factors with a LiNGAM path to BFT with probability ≥ 0.8. Majority-class baseline included |
| 07 | Table VIII | Sensitivity of the binary task: thresholds 1, 2, 3, 4, 5, 7, 9, 14, 30 days, with and without the IQR outlier filter |
| 08 | — | Original protocol behind the first manuscript version (see below), in `output/original_protocol/tables/` |
| 09 | — (not reported in the manuscript) | Transfer experiment (`TRANSFER_PROTOCOL.md`, fixed before it was run): train on Core and test on Product and the reverse, and train on the oldest 80 % and test on the newest 20 % of each scope; 4 classes and binary. `table_IX_transfer_summary` evaluates the two hypotheses |
| — | — | `python refine_lingam_bn.py`: 5-fold CV on the training splits of 48 LiNGAM-BN variants (and two hill-climbing hybrids), which chose the refined LiNGAM-BN configuration (`refinement_cv_variants`, `refinement_cv_ranking`) |

Every model in steps 05-07 is trained on the training split only (including the LiNGAM structure of LiNGAM-BN), and is evaluated once on the test split.

## Predictors

`working_time` (total resolution time, creation to resolution) always contains `fixing_time`, so using it to predict the BFT class leaks the target. It is excluded from the prediction models. `--include-working-time` reproduces the leaky setting for comparison; the LiNGAM causal analysis (steps 03-04) uses all original columns in both cases.

## Original protocol (step 08)

The first manuscript version reported BN / LiNGAM-BN results produced by `exec/bn_t.py`. `paper_protocol.py` re-implements it, and `output/original_protocol/tables/07_paper_vs_computed.csv` shows that it reproduces every value of that version (RF / LR there are this package's reconstruction; no RF / LR code survives). It differs from steps 05-07 in three ways:

1. The BN and LiNGAM-BN are fitted on **all** reports, so the 20 % test reports are also in their training data.
2. `working_time` (binned at the same 1/7/30-day cut points as BFT) is part of the evidence.
3. The binary task uses thresholds of 9 days (Core) and 4 days (Product), which are hard-coded in `exec/lingam_s.py` and have no surviving derivation.

`08_protocol_diagnostics.csv` shows the effect of conditions 1 and 2:

| Model | Fitted on | working_time | Core acc. | Product acc. |
|---|---|---|---|---|
| BN | all reports | yes | 0.779 | 0.789 |
| LiNGAM-BN | all reports | yes | **0.841** | **0.817** |
| BN | training split | yes | 0.773 | 0.789 |
| LiNGAM-BN | training split | yes | 0.706 | 0.720 |
| BN | training split | no | 0.380 | 0.402 |
| LiNGAM-BN | training split | no | 0.338 | 0.332 |

LiNGAM-BN is highest only when the test reports are part of its training data. (The BN rows fitted on the training split re-learn their structure with bnlearn, so they can differ by about 0.01 between runs.)

## Reproducibility status

`python check_manuscript.py --section <path to the manuscript's section/ folder>` (run automatically by `analyze.py` when that folder is next to this package; the manuscript source is not part of this repository) reads every value printed in the manuscript's tables, and the report counts quoted in the Data section, directly from the LaTeX source. It checks each against the computed value, to the precision the paper prints. Details are written to `output/manuscript_check.csv`.

| Manuscript table | Package output |
|---|---|
| Table I: BFT distribution | `table_I_*` |
| Data section counts (Fig. 2, filter funnel) | `figure_2_data_overview`, `01_data_collection` |
| Fig. 5, 6: causal-strength heatmaps | `figure_5/6_*` |
| Table III: strength and direction | `table_III_*` |
| Table IV: P / R / F1 (Product + Core) | `table_IV_*` |
| Table V: macro-F1 | `table_V_*` |
| Table VI: F1, Cliff's delta, p | `table_VI_*` |
| Table VII: binary classification (7 days) | `table_VII_*` |
| Table VIII: threshold and outlier sensitivity | `table_VIII_*` |

The LiNGAM bootstrap is not bit-reproducible: `DirectLiNGAM.bootstrap` resamples without a fixed seed, so a retrained run differs slightly from `data/lingam/` and `data/lingam_train/`. The figures, Table III and the prediction steps therefore use the stored outputs.
