"""RQ1b: reproduction of the cost-sensitive Transformer of Beikmohammadi et al. on the APS data (Cluster Computing,
2026), which reports a test cost of 3,440.

preprocessing_2026.py                      the authors' preprocessing (drop attributes with >= 10 % missing values,
                                           BayesianRidge iterative imputation, SVM-SMOTE, RENN, min-max scaling), with
                                           the published order and one with a validation split of real rows
                                           ("corrected")
transformer_2026_pytorch_port.py           the network ported to PyTorch, with the authors' focal loss and Keras-style
                                           initialisation
run_transformer_2026_port.py               trains the port under either protocol, with ablations of the resampling
evaluate_port_checkpoint.py                scores a saved state of a port run that was stopped early
run_tree_models_on_2026_preprocessing.py   the thesis's tree models on the same preprocessing, under the common
                                           evaluation protocol
hard_negative_check_retrained_models.py    the hard-negative comparison for the models retrained with the released code
authors_code/                              re-scoring of the released models and retraining with the released code
                                           (see its README.md)
"""
