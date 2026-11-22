# RQ1b: the PyTorch port of the cost-sensitive Transformer of Beikmohammadi et al.

The network re-implemented in PyTorch (transformer_2026_pytorch_port.py) and trained by
run_transformer_2026_port.py for 500 epochs per seed (10 imputer rounds): under the protocol with a validation
split of real rows (run names corrected_protocol*: 10 % of the real training rows held out for validation first,
imputation, resampling and scaling fitted on the other 90 %) and under the published protocol (the whole
training table imputed, resampled and scaled, then 10 % of the resampled table held out for validation), the
latter also without RENN, without SVM-SMOTE and without any resampling. The best checkpoint is the epoch with
the lowest validation cost at 0.5. (a) The extended run under the protocol with a validation split of real rows
continued seed 0 towards 4,000 epochs, was stopped by the compute budget and was scored afterwards from its last
saved checkpoint by evaluate_port_checkpoint.py. Sources:
rq1_transformer_2026_reproduction/pytorch_port/<run>_seed<k>.json.

| Variant | Seed | Epochs | Best epoch | Test, best checkpoint at 0.5 (FN / FP) | At validation threshold | Last epoch at 0.5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Protocol with a validation split of real rows | 0 | 500 | 138 | 12,540 (17 / 404) | 13,020 (18 / 402) | 15,880 |
| Protocol with a validation split of real rows | 1 | 500 | 353 | 11,100 (15 / 360) | 11,820 (17 / 332) | 10,130 |
| Protocol with a validation split of real rows | 2 | 500 | 6 | 17,270 (26 / 427) | 16,810 (24 / 481) | 14,740 |
| Protocol with a validation split of real rows, extended (stopped) | 0 | 3,250 (a) | 825 | 15,100 (25 / 260) | 14,480 (23 / 298) | 17,150 |
| Published protocol: imputation, SVM-SMOTE, RENN | 0 | 500 | 391 | 11,230 (15 / 373) | 12,320 (8 / 832) | 14,320 |
| Published protocol: imputation, SVM-SMOTE, RENN | 1 | 500 | 476 | 12,420 (17 / 392) | 13,170 (10 / 817) | 14,780 |
| Published protocol without RENN | 0 | 500 | 213 | 15,390 (22 / 439) | 18,890 (11 / 1,339) | 25,640 |
| Published protocol without RENN | 1 | 500 | 233 | 16,820 (27 / 332) | 22,880 (9 / 1,838) | 20,000 |
| Published protocol without SVM-SMOTE | 0 | 500 | 59 | 30,210 (57 / 171) | 14,970 (15 / 747) | 30,040 |
| Published protocol without SVM-SMOTE | 1 | 500 | 480 | 30,340 (58 / 134) | 14,630 (18 / 563) | 31,640 |
| Published protocol without resampling | 0 | 500 | 125 | 34,370 (67 / 87) | 15,670 (22 / 467) | 54,850 |
| Published protocol without resampling | 1 | 500 | 421 | 38,200 (75 / 70) | 13,380 (13 / 688) | 43,510 |

Columns:

- Variant: protocol and ablation (CSV: run_name, the file name stem, and label).
- Epochs, Best epoch: epochs trained and the epoch of the best validation cost (CSV: epochs, best_epoch, evaluated_from_checkpoint).
- Test, best checkpoint at 0.5 (FN / FP): test cost of the best-validation checkpoint at the 0.5 cut-off (CSV: test_*_best_checkpoint; its bootstrap 95 % interval is test_cost_interval_low/high).
- At validation threshold: the same checkpoint at the threshold that minimises the validation cost (CSV: test_*_at_validation_threshold).
- Last epoch at 0.5: the weights of the last epoch (or last saved checkpoint) at 0.5 (CSV: test_cost_last_epoch).

Unrounded values: `rq1_transformer_2026_pytorch_port_runs.csv` in this folder.
