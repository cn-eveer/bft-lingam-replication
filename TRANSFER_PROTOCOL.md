# Transfer experiment: protocol (fixed before running, 2026-10-08)

## Hypothesis

A causally constrained predictor (LiNGAM-BN) is more robust than correlation-based
predictors when the data distribution changes, i.e. it loses less performance when it is
applied to another scope or a later period than the one it was trained on.

## Models

RF, LR, BN (hill-climbing, BDeu) and refined LiNGAM-BN, with the configuration chosen
before this experiment by 5-fold cross-validation on the in-scope training split
(direct LiNGAM edges, temporal prior knowledge, total resolution time excluded,
bootstrap probability >= 0.8, at most 3 parents per node, BDeu ESS 5). No setting is
changed for the transfer experiment.

(Amended 2026-10-08, before the transfer experiment was run: ESS 5 instead of 10. ESS 1, 5
and 10 have identical CV scores for this configuration; the tie is broken by the ESS of the
standard BN, see refine_lingam_bn.py.) The original LiNGAM-BN (total-effect edges) is
reported for reference.

## Transfer settings

| Setting | Training data | Test data |
|---|---|---|
| Core -> Product | all Core reports | all Product reports |
| Product -> Core | all Product reports | all Core reports |
| Core, temporal | oldest 80 % of Core reports (by creation time) | newest 20 % |
| Product, temporal | oldest 80 % of Product reports | newest 20 % |

Each setting is run for the 4-class task and the binary task (<= 7 days). The LiNGAM
structure is learned on the training data of each setting only. The in-distribution
reference is the result of the same model on the random stratified 80/20 split of the
test scope (steps 05 and 06).

## Measures

- Macro-F1 on the transfer test data (primary) and accuracy.
- Drop = in-distribution macro-F1 - transfer macro-F1.
- 95 % bootstrap confidence interval (1,000 resamples of the test reports) of the
  difference in transfer macro-F1, LiNGAM-BN - RF.

## Success criteria

- H1 (robustness) is supported if refined LiNGAM-BN has a smaller drop than RF in at
  least 6 of the 8 setting-task combinations.
- H2 (accuracy under shift) is supported in a combination if the 95 % CI of
  LiNGAM-BN - RF lies above 0.

Results are reported in full, whether or not the hypotheses are supported.
