"""The common evaluation protocol of the thesis, implemented once here and used by every experiment.

    paths.py                          where the data, the third-party code and every result file live
    data_loading.py                   reading the two UCI files; the misclassification cost (500 per missed failure,
                                      10 per false alarm); the data audit (python -m experiments.common.data_loading)
    cost_and_evaluation.py            the cost curve, the cost-minimising threshold, stratified cross-validation with
                                      out-of-fold predictions, and the bootstrap interval of a test cost
    classifiers.py                    factories for every classifier, the tuned LightGBM settings, and the ensemble
                                      that averages LightGBM, XGBoost and CatBoost in log-odds
    imputation_and_oversampling.py    median imputation and SMOTE-family oversampling, fitted inside training folds
    synthetic_data_diagnostics.py     label-free statistics that compare synthetic failure rows with real ones
"""
