"""RQ1a: reproduction of the conditional-GAN augmentation of the APS data by Ranasinghe et al. (Ranasinghe and
Parlikad, ICPHM 2019; Ranasinghe, Lindgren, Girolami and Parlikad, IEEE Access 2019).

conditional_gan_2019.py           the generator: a conditional GAN trained on the failure rows, conditioned on their
                                  k-means cluster (k = 2), with the settings of the notebook attributed to the study
run_synthetic_row_count_sweep.py  the cost as a function of the number of synthetic rows (250 to 5,000), with the count
                                  chosen out of fold and, as a diagnostic, on the test set

The arms that add 2,000 generated rows under the common evaluation protocol (and the duplication control) are defined in
experiments/experiment_arms.py and run with experiments/run_experiment_arm.py.
"""
