# Data

The experiments use the public dataset *APS Failure at Scania Trucks* from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks

The data files are not included in this repository. Download the dataset from the page above, unpack it, and put
the two CSV files here with their original names:

```
data/aps_failure_training_set.csv    60,000 rows, 1,000 of them failures
data/aps_failure_test_set.csv        16,000 rows, 375 of them failures
```

Both files start with a 20-line licence notice (the data are released by Scania CV AB under the GNU General
Public License, version 3) before the header line. You do not have to remove it: `experiments/common/data_loading.py`
finds the header line by itself. Missing values are written as `na` in the files and become `NaN` when loaded; the
class column (`pos` = APS failure, `neg` = other) becomes 1 and 0.

To check that you have exactly the files used in the thesis, run from the repository root

```
python -m experiments.common.data_loading
```

It prints a short audit (row counts, missing-value rates, constant columns, duplicates between the two files) and
writes it to `results/data_audit.json`. The SHA-256 checksums in that file must be

```
aps_failure_training_set.csv  bb484302e3a3a1c8ef5e1f0129c4dc7cbd58f350867f95b575461ca21ab6b9da
aps_failure_test_set.csv      2cdf6f7661c7b4c63333c93cdec36a3a82350176b604a2312cf82799fb2712f3
```

(`sha256sum data/*.csv` on Linux, `shasum -a 256 data/*.csv` on macOS, `certutil -hashfile <file> SHA256` on Windows.)

The 170 attributes are anonymised by Scania. Their names have the form `xx_000`: single counters such as `aa_000`,
and histograms whose bins share a prefix, such as `ag_000` to `ag_009`. One attribute, `cd_000`, has the same value
in every training row and is dropped by every experiment.
