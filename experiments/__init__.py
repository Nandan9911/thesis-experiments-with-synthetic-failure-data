"""RQ2: a survey of other ways to create synthetic failure rows, or to use the APS data better.

The generators of this package (Gaussian copula, TVAE, CTGAN, TabDDPM, CTAB-GAN+, GReaT, Forest-Flow and TabEBM)
are wrapped as augmenters fitted inside each training fold and evaluated as arms of experiments/experiment_arms.py
with 2,000 synthetic failure rows each; tabicl_classifier.py provides the in-context classifier TabICLv2.
run_synthetic_data_diagnostics.py measures how realistic and how close to copies the rows of every generator are,
and pilots/ holds smaller runs that measured time, memory and first results on reduced settings.
"""
