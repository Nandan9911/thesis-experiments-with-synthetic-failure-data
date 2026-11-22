# Data audit of the two UCI files

Integrity checks of aps_failure_training_set.csv and aps_failure_test_set.csv ('na' read as missing),
written by `python -m experiments.common.data_loading` to results/data_audit.json. The two costs at the
end are the cost measure (500 per missed failure, 10 per false alarm) of the trivial rules.

| Quantity | Value |
| :--- | ---: |
| Training rows | 60,000 |
| Failures among the training rows | 1,000 |
| Test rows | 16,000 |
| Failures among the test rows | 375 |
| Attributes | 170 |
| Same attributes, in the same order, in both files | yes |
| Share of missing cells, training file | 0.0833 |
| Share of missing cells, test file | 0.0841 |
| Share of missing cells in failure rows (training) | 0.1715 |
| Share of missing cells in healthy rows (training) | 0.0818 |
| Attributes with more than 50 % missing (training) | 8 |
| Attributes with more than 70 % missing (training) | 7 |
| Constant attributes (training) | cd_000 |
| Test rows identical to a training row (values rounded to 3 decimals) | 0 |
| Test cost when nothing is flagged | 187,500 |
| Test cost when everything is flagged | 156,250 |
| SHA-256 of the training file | bb484302e3a3a1c8ef5e1f0129c4dc7cbd58f350867f95b575461ca21ab6b9da |
| SHA-256 of the test file | 2cdf6f7661c7b4c63333c93cdec36a3a82350176b604a2312cf82799fb2712f3 |

Columns:

- Value: the value in data_audit.json (CSV: value; field is its key in the file).

Unrounded values: `data_audit.csv` in this folder.
