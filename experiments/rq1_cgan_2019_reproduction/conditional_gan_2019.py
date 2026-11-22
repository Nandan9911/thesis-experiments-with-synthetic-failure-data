"""The conditional GAN of Ranasinghe et al. for APS failure rows, re-implemented in PyTorch (RQ1a).

Research question 1 of the thesis reproduces two published results on the Scania APS data. This module belongs to
the first, Ranasinghe and Parlikad (ICPHM 2019) and its journal version, Ranasinghe, Lindgren, Girolami and Parlikad
(IEEE Access 2019), who trained a conditional generative adversarial network (cGAN) on the failure rows of the training
file and appended 2,000 generated failure rows to the 60,000 training rows before training their classifiers.

A GAN trains two networks against each other: a generator, which turns random noise into rows, and a discriminator,
which learns to tell real rows from generated ones; the generator is updated so that its rows are classified as
real. In a *conditional* GAN both networks also receive a condition vector, so that rows of a chosen kind can be
requested. In the study of Ranasinghe et al. the condition is the cluster of the failure row: the failure rows are split
into two groups with k-means (k = 2), and the one-hot cluster label is the condition.

The papers do not give layer sizes or training settings. They are taken from a Colab notebook attributed to the study,
and this implementation keeps them: one hidden layer of 256 ReLU units in both networks, a 100-dimensional noise vector
drawn uniformly from [-1, 1], Xavier initialisation, sigmoid outputs, Adam with learning rate 2e-4, batches of 100 rows
and 30,000 training steps, missing values replaced by 0 before scaling and clustering, and the same number of generated
rows from each cluster.

By default two details of the notebook are changed (the notebook's versions remain available):

* Scaling. The notebook standardises the attributes (StandardScaler) but ends the generator in a sigmoid, whose
  output lies in (0, 1). After the inverse transform every generated value therefore lies between the attribute's
  mean and one standard deviation above it; values below the mean cannot be produced at all. The default here,
  scaling="minmax", scales each attribute to [0, 1], a range the sigmoid covers. scaling="standard" is the notebook.
* Loss. The notebook's discriminator ends in a sigmoid, and its loss is a sigmoid cross-entropy computed "from
  logits", so the sigmoid is applied twice. The effective probability sigmoid(sigmoid(a)) can only lie between 0.5
  and 0.73, so the loss never approaches zero and its gradients are damped. The default trains the discriminator on
  logits with a single sigmoid inside the loss. literal_loss=True reproduces the double sigmoid and also the
  notebook's way of drawing a batch, in which the cluster labels are drawn independently of the real rows they
  accompany.

The arm random_forest_with_conditional_gan_2019_literal_notebook_rows of experiment_arms.py runs the notebook's
versions.

The study fitted the GAN once, on all 1,000 failure rows of the training file. Here it is fitted inside each
training fold of the cross-validation, on that fold's failure rows only (conditional_gan_augmenter below), so that
no row of the validation fold influences the rows that are added (the common evaluation protocol). The scaler and
the k-means clustering are part of the fitted generator and are therefore also fitted on the fold's failure rows
only.

The notebook trains on rows whose missing values were set to 0, so the generator produces complete rows. The tuned
LightGBM, the main classifier of the thesis, is given the missing values themselves (it handles them natively), and
complete synthetic rows would differ from the real failures by that alone. sample(..., reapply_missingness=True)
therefore blanks each generated cell independently, with the probability that the same attribute is missing among
the real failure rows the generator was fitted on. This matches the missing rate of each attribute, not the pattern
of which attributes tend to be missing together in one row. In the arms with median imputation the generator is
fitted on rows that are already complete, so there is nothing to put back.

Training 30,000 steps on the roughly 800 failure rows of one training fold takes a few minutes on one CPU thread.
The module has no command line; it is used by experiments/experiment_arms.py (the arms with "conditional_gan_2019"
in their name), by run_synthetic_row_count_sweep.py in this folder, and by the RQ2 diagnostics and the RQ3 cold-start
study.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class _OneHiddenLayerNetwork(nn.Module):
    """Linear -> ReLU -> Linear, optionally followed by an output activation (both networks of the notebook)."""

    def __init__(self, input_size, hidden_units, output_size, output_activation):
        super().__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_units)
        self.output_layer = nn.Linear(hidden_units, output_size)
        self.output_activation = output_activation
        # Xavier (Glorot) normal weights and zero biases, as in the notebook. The two layers are created first and
        # initialised afterwards; this order fixes the sequence of random draws and is kept for reproducibility.
        for layer in (self.hidden_layer, self.output_layer):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        hidden = torch.relu(self.hidden_layer(inputs))
        outputs = self.output_layer(hidden)
        return self.output_activation(outputs) if self.output_activation is not None else outputs


class ConditionalGan2019:
    """Conditional GAN for failure rows, with the architecture and training settings of the notebook.

    The keyword parameters steps, seed, scaling and literal_loss are used by the other experiment scripts. The
    remaining ones keep the notebook's values by default: noise_dim (length of the noise vector), hidden_units (width
    of the hidden layer of both networks), learning_rate (Adam), batch_size, n_clusters (k of the k-means condition),
    noise_distribution ("uniform" on [-1, 1], otherwise standard normal) and missing_value_fill (the value that
    replaces missing cells before scaling).

    fit(X_failures) trains on a DataFrame of failure rows; sample(n) returns n generated rows as a DataFrame with the
    same columns. Each logged step (every 1,000th and the last) appends (step, discriminator loss, generator loss)
    to loss_history.
    """

    def __init__(self, noise_dim=100, hidden_units=256, learning_rate=2e-4, batch_size=100, steps=30000,
                 n_clusters=2, scaling="minmax", literal_loss=False, noise_distribution="uniform",
                 missing_value_fill=0.0, seed=0, device="cpu", verbose=False):
        self.noise_dim, self.hidden_units, self.learning_rate = noise_dim, hidden_units, learning_rate
        self.batch_size, self.steps, self.n_clusters = batch_size, steps, n_clusters
        self.scaling, self.literal_loss, self.noise_distribution = scaling, literal_loss, noise_distribution
        self.missing_value_fill, self.seed, self.device, self.verbose = missing_value_fill, seed, device, verbose
        self.loss_history = []

    def _noise(self, n_rows):
        """Noise for training, drawn from torch's global generator (seeded in fit)."""
        if self.noise_distribution == "uniform":
            return torch.rand(n_rows, self.noise_dim, device=self.device) * 2 - 1
        return torch.randn(n_rows, self.noise_dim, device=self.device)

    def fit(self, X_failures: pd.DataFrame):
        torch.set_num_threads(1)  # tiny networks: one thread is fastest and leaves the other cores to the classifier
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.columns = list(X_failures.columns)
        values = X_failures.to_numpy(dtype=np.float64)
        # per-attribute missing rate of the failure rows, put back into the generated rows by sample()
        self.missing_fraction_per_column_ = np.isnan(values).mean(axis=0)
        values = np.where(np.isnan(values), self.missing_value_fill, values)
        self.scaler = MinMaxScaler() if self.scaling == "minmax" else StandardScaler()
        scaled = self.scaler.fit_transform(values)
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.seed).fit(scaled)
        cluster_labels = self.kmeans.labels_
        self.cluster_frequencies_ = np.bincount(cluster_labels, minlength=self.n_clusters) / len(cluster_labels)
        one_hot_labels = np.eye(self.n_clusters)[cluster_labels]
        rows = torch.tensor(scaled, dtype=torch.float32, device=self.device)
        conditions = torch.tensor(one_hot_labels, dtype=torch.float32, device=self.device)
        n_attributes = scaled.shape[1]
        self.generator = _OneHiddenLayerNetwork(self.noise_dim + self.n_clusters, self.hidden_units, n_attributes,
                                                torch.sigmoid).to(self.device)
        # With literal_loss the discriminator ends in a sigmoid although the loss below expects logits (the
        # notebook's double sigmoid); otherwise it outputs logits.
        self.discriminator = _OneHiddenLayerNetwork(n_attributes + self.n_clusters, self.hidden_units, 1,
                                                    torch.sigmoid if self.literal_loss else None).to(self.device)
        generator_optimiser = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        discriminator_optimiser = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        bce_on_logits = nn.BCEWithLogitsLoss()
        n_rows = rows.shape[0]
        batch_size = min(self.batch_size, n_rows)
        for step in range(self.steps):
            batch_index = torch.randperm(n_rows, device=self.device)[:batch_size]
            real_batch, condition_batch = rows[batch_index], conditions[batch_index]
            if self.literal_loss:  # the notebook draws the label batch independently of the row batch
                label_index = torch.randperm(n_rows, device=self.device)[:batch_size]
                condition_batch = conditions[label_index]
            noise = self._noise(batch_size)
            # discriminator step: real rows labelled 1, generated rows labelled 0
            generated = self.generator(torch.cat([noise, condition_batch], 1)).detach()
            score_real = self.discriminator(torch.cat([real_batch, condition_batch], 1))
            score_generated = self.discriminator(torch.cat([generated, condition_batch], 1))
            discriminator_loss = (bce_on_logits(score_real, torch.ones_like(score_real))
                                  + bce_on_logits(score_generated, torch.zeros_like(score_generated)))
            discriminator_optimiser.zero_grad()
            discriminator_loss.backward()
            discriminator_optimiser.step()
            # generator step with the same noise, as in the notebook: generated rows should be scored as real
            generated = self.generator(torch.cat([noise, condition_batch], 1))
            score_generated = self.discriminator(torch.cat([generated, condition_batch], 1))
            generator_loss = bce_on_logits(score_generated, torch.ones_like(score_generated))
            generator_optimiser.zero_grad()
            generator_loss.backward()
            generator_optimiser.step()
            if step % 1000 == 0 or step == self.steps - 1:
                self.loss_history.append((step, discriminator_loss.item(), generator_loss.item()))
                if self.verbose:
                    print(f"    cGAN step {step}: discriminator loss {discriminator_loss.item():.3f}, "
                          f"generator loss {generator_loss.item():.3f}", flush=True)
        return self

    @torch.no_grad()
    def sample(self, n: int, reapply_missingness: bool = False, seed: int | None = None,
               alternate_clusters: bool = False) -> pd.DataFrame:
        """n generated failure rows in the original units (float32 DataFrame with the training columns).

        The rows are split as evenly as possible between the clusters and returned in cluster order: first all rows
        of cluster 0, then all rows of cluster 1. With alternate_clusters, the clusters alternate row by row instead
        (cluster 0, cluster 1, cluster 0, ...), so that the first k rows of the result contain the same number of
        rows from each cluster, to within one; the synthetic-row count sweep, which takes the first rows of one
        pool for every count, uses this. With reapply_missingness, cells are blanked as described in the module
        docstring. seed (default: the generator's own seed) fixes the noise and the missingness draws.
        """
        sampling_seed = self.seed if seed is None else seed
        noise_generator = torch.Generator(device=self.device).manual_seed(sampling_seed)
        rng = np.random.default_rng(sampling_seed)
        # the same number of rows from each cluster, as in the notebook (125 of each per block of 250 rows)
        rows_per_cluster = [n // self.n_clusters + (1 if cluster < n % self.n_clusters else 0)
                            for cluster in range(self.n_clusters)]
        generated_blocks = []
        for cluster, n_cluster_rows in enumerate(rows_per_cluster):
            condition = torch.zeros(n_cluster_rows, self.n_clusters, device=self.device)
            condition[:, cluster] = 1
            if self.noise_distribution == "uniform":
                noise = (torch.rand(n_cluster_rows, self.noise_dim, generator=noise_generator, device=self.device)
                         * 2 - 1)
            else:
                noise = torch.randn(n_cluster_rows, self.noise_dim, generator=noise_generator, device=self.device)
            generated_blocks.append(self.generator(torch.cat([noise, condition], 1)).cpu().numpy())
        values = self.scaler.inverse_transform(np.vstack(generated_blocks))
        if alternate_clusters:
            cluster_of_row = np.repeat(np.arange(self.n_clusters), rows_per_cluster)
            position_in_cluster = np.concatenate([np.arange(size) for size in rows_per_cluster])
            # np.lexsort sorts by its last key first: by position within the cluster, then by cluster
            values = values[np.lexsort((cluster_of_row, position_in_cluster))]
        if reapply_missingness:
            blank = rng.random(values.shape) < self.missing_fraction_per_column_[None, :]
            values = np.where(blank, np.nan, values)
        return pd.DataFrame(values.astype(np.float32), columns=self.columns)


def conditional_gan_augmenter(n_synthetic=2000, reapply_missingness=False, steps=30000, scaling="minmax",
                              literal_loss=False, **gan_options):
    """Augmenter for run_cross_validation: augment(X_train, y_train, seed) -> (X_augmented, y_augmented).

    Inside each training fold it fits a ConditionalGan2019 (with the given seed) on the fold's real failure rows
    only and appends n_synthetic generated rows, labelled as failures, to the fold's training rows. The validation
    fold is never seen by the generator. Further keyword arguments go to ConditionalGan2019.
    """
    def augment(X_train: pd.DataFrame, y_train: np.ndarray, seed: int):
        gan = ConditionalGan2019(seed=seed, steps=steps, scaling=scaling, literal_loss=literal_loss,
                                 **gan_options).fit(X_train[y_train == 1])
        X_synthetic = gan.sample(n_synthetic, reapply_missingness=reapply_missingness, seed=seed + 1)
        X_augmented = pd.concat([X_train, X_synthetic], ignore_index=True)
        y_augmented = np.concatenate([y_train, np.ones(len(X_synthetic), dtype=y_train.dtype)])
        return X_augmented, y_augmented
    return augment
