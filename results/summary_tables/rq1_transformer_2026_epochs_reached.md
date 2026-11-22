# RQ1b: epochs reached by the training runs

The number of epochs that each training run of RQ1b reached. For the runs of the released training code
from scratch (the retraining runs, the test-inclusion controls and the two verification reruns) it is the epoch
in the run's state.json; for the PyTorch port it is epochs_run in the run's result file, except for the
extended run under the protocol with a validation split of real rows, which was stopped and scored afterwards
from its last saved checkpoint: there it is the saved epoch plus one, since the checkpoint counts epochs from 0.
The number of epochs each run was planned to reach is not stored in the result files and is not shown. Sources,
in results/rq1_transformer_2026_reproduction/: authors_code/retraining_runs/<run>/state.json and
pytorch_port/<run>_seed<k>.json.

## Retraining with the released code

| Run | Seed | Epochs reached |
| :--- | ---: | ---: |
| Retraining | 0 | 8,000 |
| Retraining | 1 | 5,950 |
| Retraining | 2 | 1,000 |
| Retraining | 3 | 1,000 |
| Retraining | 4 | 1,000 |
| Half of the test rows in training | 0 | 8,000 |
| Half of the test rows in training | 1 | 1,000 |
| Half of the test rows in training | 2 | 1,000 |
| All test rows in training | 0 | 1,000 |
| Retraining, verification rerun | 0 | 300 |
| Half of the test rows in training, verification rerun | 0 | 300 |

## The PyTorch port

| Run | Seed | Epochs reached |
| :--- | ---: | ---: |
| Protocol with a validation split of real rows | 0 | 500 |
| Protocol with a validation split of real rows | 1 | 500 |
| Protocol with a validation split of real rows | 2 | 500 |
| Protocol with a validation split of real rows, extended (stopped) | 0 | 3,250 |
| Published protocol: imputation, SVM-SMOTE, RENN | 0 | 500 |
| Published protocol: imputation, SVM-SMOTE, RENN | 1 | 500 |
| Published protocol without RENN | 0 | 500 |
| Published protocol without RENN | 1 | 500 |
| Published protocol without SVM-SMOTE | 0 | 500 |
| Published protocol without SVM-SMOTE | 1 | 500 |
| Published protocol without resampling | 0 | 500 |
| Published protocol without resampling | 1 | 500 |

Columns:

- Run: design of the run (CSV: label; run is the run name, implementation is authors_code or pytorch_port).
- Epochs reached: CSV: epochs_reached; source names the field it was read from.

Unrounded values: `rq1_transformer_2026_epochs_reached.csv` in this folder.
