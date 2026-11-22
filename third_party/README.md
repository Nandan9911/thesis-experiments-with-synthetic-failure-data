# External code

Two experiments run code written by other authors. That code is not copied into this repository; it is cloned at
the exact commit used for the thesis by

```
bash third_party/fetch_third_party_code.sh
```

which creates

| Folder | Repository | Commit | Needed for |
|---|---|---|---|
| `third_party/cost-sensitive-transformer-model/` | https://github.com/AliBeikmohammadi/Cost-Sensitive-Transformer-Model | `4db444a3958f0a5e95536e29b56206c497bb6d3d` (11 March 2026) | RQ1b: the authors' preprocessing, training code and five trained models of Beikmohammadi et al. (2026). Used by `experiments/rq1_transformer_2026_reproduction/authors_code/` and by the retraining jobs in `cloud_runs/`. |
| `third_party/CTAB-GAN-Plus/` | https://github.com/Team-TUD/CTAB-GAN-Plus | `6a6f90188cca3dac2c533fd5e8e7f20de074365b` (14 May 2024) | RQ2: the CTAB-GAN+ generator (arm `lightgbm_with_ctab_gan_plus_rows`). |

Nothing else needs these folders, so you can skip the script if you do not run those experiments. A fresh clone of
the authors' repository takes about 100 MB; it grows by about 450 MB when `prepare_authors_inputs.py` writes the
regenerated input tables into it.

If you keep the repositories somewhere else, point the code to them with the environment variables
`APS_AUTHORS_2026_REPOSITORY` and `APS_CTAB_GAN_PLUS_REPOSITORY`.

`authors_code/prepare_authors_inputs.py` writes the regenerated input tables of the authors' training code into the
authors' folder, because that is where their `TrainingCode.py` expects them. The clone is therefore not left untouched
after that script has run; `git -C third_party/cost-sensitive-transformer-model status` shows what was added.

Both folders are listed in `.gitignore`.
