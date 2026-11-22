# RQ1b: means, ranges and totals quoted in the text

Aggregates of the RQ1b tables that the thesis quotes in its running text, computed from the same files:
rq1_transformer_2026_shipped_models_rescored, _clean_retraining_runs, _retraining_and_contamination_runs and
_pytorch_port_runs. Ranges over PyTorch port runs leave out the extended run scored from a checkpoint.

| Quantity | Value |
| :--- | ---: |
| Released models, published preprocessing: mean test cost | 3,440 |
| Released models, published preprocessing: lowest test cost of a run | 3,060 |
| Released models, published preprocessing: highest test cost of a run | 3,540 |
| Released models, published preprocessing: mean missed failures | 6.8 |
| Released models, published preprocessing: mean false alarms | 4.0 |
| Released model run 1 on the 60,000 training rows: cost at 0.5 | 12,660 |
| Released model run 1 on the 60,000 training rows: missed failures | 17 |
| Released model run 1 on the 60,000 training rows: false alarms | 416 |
| Retraining runs without test rows: number of seeds | 5 |
| Retraining runs without test rows: median test cost of the final evaluation | 18,240 |
| Retraining runs without test rows: lowest test cost of the final evaluation | 17,060 |
| Retraining runs without test rows: highest test cost of the final evaluation | 18,570 |
| Retraining runs without test rows: fewest epochs | 1,000 |
| Retraining runs without test rows: most epochs | 8,000 |
| Retraining runs without test rows: estimated GPU hours, total | 114.0 |
| Retraining runs without test rows: mean test cost at 1,000 epochs | 18,090 |
| Retraining runs without test rows: lowest test cost at 1,000 epochs | 17,060 |
| Retraining runs without test rows: highest test cost at 1,000 epochs | 18,570 |
| Test-inclusion controls with half of the test rows: number of seeds | 3 |
| PyTorch port, protocol with a validation split of real rows: mean test cost, best checkpoint at 0.5 | 13,637 |
| PyTorch port, protocol with a validation split of real rows: lowest test cost, best checkpoint at 0.5 | 11,100 |
| PyTorch port, protocol with a validation split of real rows: highest test cost, best checkpoint at 0.5 | 17,270 |
| PyTorch port, protocol with a validation split of real rows: lowest test cost at the validation threshold | 11,820 |
| PyTorch port, protocol with a validation split of real rows: highest test cost at the validation threshold | 16,810 |
| PyTorch port, published protocol: imputation, SVM-SMOTE, RENN: mean test cost, best checkpoint at 0.5 | 11,825 |
| PyTorch port, published protocol: imputation, SVM-SMOTE, RENN: lowest test cost, best checkpoint at 0.5 | 11,230 |
| PyTorch port, published protocol: imputation, SVM-SMOTE, RENN: highest test cost, best checkpoint at 0.5 | 12,420 |
| PyTorch port, published protocol: imputation, SVM-SMOTE, RENN: lowest test cost at the validation threshold | 12,320 |
| PyTorch port, published protocol: imputation, SVM-SMOTE, RENN: highest test cost at the validation threshold | 13,170 |
| PyTorch port, published protocol without RENN: mean test cost, best checkpoint at 0.5 | 16,105 |
| PyTorch port, published protocol without RENN: lowest test cost, best checkpoint at 0.5 | 15,390 |
| PyTorch port, published protocol without RENN: highest test cost, best checkpoint at 0.5 | 16,820 |
| PyTorch port, published protocol without RENN: lowest test cost at the validation threshold | 18,890 |
| PyTorch port, published protocol without RENN: highest test cost at the validation threshold | 22,880 |
| PyTorch port, published protocol without SVM-SMOTE: mean test cost, best checkpoint at 0.5 | 30,275 |
| PyTorch port, published protocol without SVM-SMOTE: lowest test cost, best checkpoint at 0.5 | 30,210 |
| PyTorch port, published protocol without SVM-SMOTE: highest test cost, best checkpoint at 0.5 | 30,340 |
| PyTorch port, published protocol without SVM-SMOTE: lowest test cost at the validation threshold | 14,630 |
| PyTorch port, published protocol without SVM-SMOTE: highest test cost at the validation threshold | 14,970 |
| PyTorch port, published protocol without resampling: mean test cost, best checkpoint at 0.5 | 36,285 |
| PyTorch port, published protocol without resampling: lowest test cost, best checkpoint at 0.5 | 34,370 |
| PyTorch port, published protocol without resampling: highest test cost, best checkpoint at 0.5 | 38,200 |
| PyTorch port, published protocol without resampling: lowest test cost at the validation threshold | 13,380 |
| PyTorch port, published protocol without resampling: highest test cost at the validation threshold | 15,670 |

Unrounded values: `rq1_transformer_2026_summary_numbers.csv` in this folder.
