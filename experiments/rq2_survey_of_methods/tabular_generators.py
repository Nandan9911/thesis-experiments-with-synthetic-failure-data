"""Six generators of synthetic failure rows for the RQ2 survey, each wrapped as an augmenter fitted inside the fold.

Part of RQ2, the survey of other ways to create synthetic failure rows. The arms `lightgbm_with_<generator>_rows`
in experiments/experiment_arms.py add 2,000 rows from one of these generators to every training fold, and
rq2_survey_of_methods/run_synthetic_data_diagnostics.py measures how realistic and how close to copies the rows
are. The module is not run on its own; it is used through the arm runner, for example

    python -m experiments.run_experiment_arm --arm lightgbm_with_ctgan_rows --seeds 11 29 47 --final-test

Apart from the CTAB-GAN+ repository in third_party/ (needed by that generator only) and the pretrained language model
of GReaT (downloaded from Hugging Face on first use), it reads nothing. It writes only temporary files; the
generators return their rows as DataFrames.

Every generator follows the same contract as the conditional GAN of Ranasinghe et al., Forest-Flow and TabEBM:

    augment(X_train, y_train, seed) -> (X_augmented, y_augmented)

The generator is fitted on the failure rows of the training fold only, and the rows it samples are appended with label
1. Neither the validation fold nor the official test set ever reaches a generator, as the common evaluation protocol
requires.

Preprocessing shared by all six (class FailureRowGenerator):
  * None of these models accepts missing values, so the failure rows are median-imputed before fitting.
  * The APS attributes are non-negative counters and histogram bins with very long right tails. Most generators
    therefore work on log(1 + x), standardised to mean 0 and standard deviation 1 per attribute. Attributes that
    are constant among the fitting rows are left out of the model and put back afterwards.
  * As in the study of Ranasinghe et al., the failure rows are split into two groups by k-means clustering (k = 2) on
    the transformed attributes. The generators that can condition on a discrete column (CTGAN, CTAB-GAN+, GReaT and the
    diffusion model) are given the cluster label and draw new rows cluster by cluster, in the proportions of the real
    clusters.
  * Generated rows come out complete, but real failure rows are not: they miss about 17 % of their values, twice
    the share of healthy rows, and a classifier can use that difference. Each generated row therefore receives the
    missing-value pattern of a real failure row drawn at random (with replacement), so that synthetic and real
    failures show the same missingness.

The generators (keys of SURVEY_GENERATORS):
  gaussian_copula  Gaussian copula (copulas library). Every attribute keeps its own marginal distribution, here a
                   kernel density estimate, and the dependence between the attributes is modelled as a multivariate
                   normal distribution of their normal scores. Unconditional.
  tvae             TVAE (Xu et al., 2019; ctgan library), a variational autoencoder for tables. Continuous columns
                   are encoded by "mode-specific normalisation": a Gaussian mixture is fitted to each column and a
                   value is represented by its mixture component and its standardised position within that
                   component. Unconditional.
  ctgan            CTGAN (Xu et al., 2019; ctgan library), a conditional generative adversarial network for tables
                   with the same mode-specific normalisation. Its generator learns to produce rows for a requested
                   value of a discrete column, here the cluster label.
  tabddpm          A denoising diffusion model following the numerical part of TabDDPM (Kotelnikov et al., 2023).
                   Gaussian noise is added to the rows in 1,000 small steps, a neural network learns to predict the
                   noise that was added, and new rows are made by starting from pure noise and removing it step by
                   step. Own compact implementation (class GaussianDDPM), conditioned on the cluster label.
  ctab_gan_plus    CTAB-GAN+ (Zhao et al., 2022): CTGAN plus an auxiliary classifier whose loss keeps the generated
                   rows consistent with a target column (here the cluster label), and special encodings for
                   long-tailed and mixed columns. Runs the authors' code from third_party/CTAB-GAN-Plus.
  great            GReaT (Borisov et al., 2023): every row is written as a short sentence, a pretrained language
                   model is fine-tuned on the sentences, and new rows are generated as text and parsed back. This
                   arm failed on APS; see GReaTFailureGenerator.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from experiments.common.paths import CTAB_GAN_PLUS_REPOSITORY_DIR


class FailureRowGenerator:
    """Preprocessing and post-processing shared by the generators of this module.

    A subclass calls `_prepare` on the real failure rows before fitting its model, `_inverse` on what the model
    generates, and `_finish` to put the real missingness back. log_space and standardise switch the log(1 + x) map
    and the per-attribute standardisation on or off; n_clusters is the number of k-means clusters whose label serves
    as the conditioning variable (2, as in the study of Ranasinghe et al.).
    """

    def __init__(self, seed=0, log_space=True, standardise=True, n_clusters=2, verbose=True):
        self.seed, self.log_space, self.standardise, self.n_clusters, self.verbose = (seed, log_space, standardise,
                                                                                      n_clusters, verbose)

    # --------------------------------------------------------------------------------------- forward and inverse maps
    def _prepare(self, X_failures: pd.DataFrame) -> np.ndarray:
        """Impute, transform and cluster the real failure rows. Returns the model input, which holds only the
        attributes that vary among these rows."""
        self.columns = list(X_failures.columns)
        values = X_failures.to_numpy(dtype=np.float64)
        self.missingness_patterns_ = np.isnan(values)          # one pattern per real row, drawn again in _finish
        self.medians_ = np.nanmedian(values, axis=0)
        self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        values = np.where(np.isnan(values), self.medians_[None, :], values)
        self.shift_ = np.minimum(values.min(axis=0), 0.0)      # 0 for non-negative counters; guards log1p otherwise
        Z = np.log1p(values - self.shift_) if self.log_space else values - self.shift_
        if self.standardise:
            self.means_, self.standard_deviations_ = Z.mean(axis=0), Z.std(axis=0) + 1e-6
            Z = (Z - self.means_) / self.standard_deviations_
        self.varying_attributes_ = Z.std(axis=0) > 1e-8       # constant attributes are restored in _inverse
        self.constant_values_ = Z.mean(axis=0)
        Z_varying = Z[:, self.varying_attributes_]
        # the auxiliary label of the study of Ranasinghe et al.: k-means (k = 2) on the transformed failure rows
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.seed).fit(Z_varying)
        self.cluster_labels_ = self.kmeans_.labels_
        cluster_sizes = np.bincount(self.cluster_labels_, minlength=self.n_clusters)
        self.cluster_proportions_ = cluster_sizes / len(self.cluster_labels_)
        return Z_varying

    def _inverse(self, Z_varying: np.ndarray) -> np.ndarray:
        """Map generated rows back to counter units: constant attributes restored, negative values set to 0."""
        Z = np.tile(self.constant_values_, (len(Z_varying), 1))
        Z[:, self.varying_attributes_] = Z_varying
        if self.standardise:
            Z = Z * self.standard_deviations_ + self.means_
        values = (np.expm1(Z) if self.log_space else Z) + self.shift_
        return np.maximum(values, 0.0)

    def _finish(self, values: np.ndarray, reapply_missingness: bool, seed: int) -> pd.DataFrame:
        """Optionally give every generated row the missing-value pattern of a randomly drawn real failure row, and
        return the rows as a float32 DataFrame with the original column names."""
        if reapply_missingness:
            rng = np.random.default_rng(seed)
            patterns = self.missingness_patterns_[rng.integers(0, len(self.missingness_patterns_), len(values))]
            values = np.where(patterns, np.nan, values)
        return pd.DataFrame(values.astype(np.float32), columns=self.columns)

    def _draw_cluster_labels(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Cluster labels for n new rows, drawn in the proportions of the real clusters."""
        return rng.choice(self.n_clusters, size=n, p=self.cluster_proportions_)


# ---------------------------------------------------------------------------------------------------- Gaussian copula
class GaussianCopulaFailureGenerator(FailureRowGenerator):
    """Gaussian copula with kernel-density marginals (copulas.multivariate.GaussianMultivariate). Unconditional."""

    def fit(self, X_failures):
        from copulas.multivariate import GaussianMultivariate
        from copulas.univariate import GaussianKDE
        started = time.time()
        Z = self._prepare(X_failures)
        self.model_ = GaussianMultivariate(distribution=GaussianKDE, random_state=self.seed)
        self.model_.fit(pd.DataFrame(Z))
        if self.verbose:
            print(f"    Gaussian copula fit on {Z.shape} in {time.time() - started:.0f}s", flush=True)
        return self

    def sample(self, n, reapply_missingness=True, seed=None):
        seed = self.seed if seed is None else seed
        self.model_.set_random_state(seed)
        Z = self.model_.sample(n).to_numpy(dtype=np.float64)
        return self._finish(self._inverse(Z), reapply_missingness, seed)


# -------------------------------------------------------------------------------------------------------- TVAE, CTGAN
class TVAEFailureGenerator(FailureRowGenerator):
    """TVAE from the ctgan library, trained for `epochs` passes over the failure rows. Unconditional."""

    def __init__(self, epochs=300, batch_size=200, **kwargs):
        super().__init__(**kwargs)
        self.epochs, self.batch_size = epochs, batch_size

    def fit(self, X_failures):
        from ctgan import TVAE
        import torch
        torch.manual_seed(self.seed)
        started = time.time()
        Z = self._prepare(X_failures)
        batch_size = min(self.batch_size, len(Z))
        self.model_ = TVAE(epochs=self.epochs, batch_size=batch_size, enable_gpu=torch.cuda.is_available(),
                           verbose=False)
        self.model_.fit(pd.DataFrame(Z, columns=[f"f{i}" for i in range(Z.shape[1])]))
        if self.verbose:
            print(f"    TVAE fit ({self.epochs} epochs) on {Z.shape} in {time.time() - started:.0f}s", flush=True)
        return self

    def sample(self, n, reapply_missingness=True, seed=None):
        seed = self.seed if seed is None else seed
        self.model_.set_random_state(seed)
        Z = self.model_.sample(n).to_numpy(dtype=np.float64)
        return self._finish(self._inverse(Z), reapply_missingness, seed)


class CTGANFailureGenerator(FailureRowGenerator):
    """CTGAN from the ctgan library, with the cluster label as its discrete conditioning column."""

    def __init__(self, epochs=300, batch_size=200, **kwargs):
        super().__init__(**kwargs)
        self.epochs, self.batch_size = epochs, batch_size

    def fit(self, X_failures):
        from ctgan import CTGAN
        import torch
        # Only PyTorch is seeded here. The ctgan library also draws the conditional vectors of its training batches
        # from NumPy's global random generator, which is left unseeded, so two fits on the same rows are not identical.
        torch.manual_seed(self.seed)
        started = time.time()
        Z = self._prepare(X_failures)
        # CTGAN's discriminator judges rows in packs of 10 ("pac"), so the batch size must be a multiple of 10
        batch_size = max(10, (min(self.batch_size, len(Z)) // 10) * 10)
        table = pd.DataFrame(Z, columns=[f"f{i}" for i in range(Z.shape[1])])
        table["aux_cluster"] = pd.Series(self.cluster_labels_).astype(str)
        self.model_ = CTGAN(epochs=self.epochs, batch_size=batch_size, enable_gpu=torch.cuda.is_available(),
                            verbose=False)
        self.model_.fit(table, discrete_columns=["aux_cluster"])
        if self.verbose:
            print(f"    CTGAN fit ({self.epochs} epochs) on {Z.shape} in {time.time() - started:.0f}s", flush=True)
        return self

    def sample(self, n, reapply_missingness=True, seed=None):
        seed = self.seed if seed is None else seed
        self.model_.set_random_state(seed)
        rng = np.random.default_rng(seed)
        rows_per_cluster = np.bincount(self._draw_cluster_labels(n, rng), minlength=self.n_clusters)
        parts = [self.model_.sample(int(count), condition_column="aux_cluster", condition_value=str(cluster))
                 for cluster, count in enumerate(rows_per_cluster) if count > 0]
        Z = pd.concat(parts, ignore_index=True).drop(columns=["aux_cluster"]).to_numpy(dtype=np.float64)
        return self._finish(self._inverse(Z), reapply_missingness, seed)


# ----------------------------------------------------------------------------------------- Gaussian tabular diffusion
class GaussianDDPM:
    """A compact class-conditional denoising diffusion probabilistic model (DDPM) for numerical tables.

    It follows the numerical branch of TabDDPM (Kotelnikov, Baranchuk, Rubachev and Babenko, ICML 2023). A row x_0 is
    noised in diffusion_steps = 1,000 steps with a linear schedule of noise variances beta_t from 1e-4 to 0.02, so
    that after t steps x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise, where alpha_bar_t is the
    product of (1 - beta) over the first t steps. An MLP (the denoiser) receives x_t, a sinusoidal embedding of t and
    a learned embedding of the class label, and is trained to predict the noise. Sampling starts from pure noise and
    applies the learned denoising step 1,000 times ("ancestral sampling"). TabDDPM's multinomial branch for
    categorical columns is not needed on APS, whose attributes are all numerical.
    """

    def __init__(self, n_attributes, n_classes=2, hidden_layers=(512, 512, 512), diffusion_steps=1000, steps=20000,
                 batch_size=256, learning_rate=1e-3, seed=0, device="cpu"):
        import torch
        import torch.nn as nn
        self.n_attributes, self.n_classes = n_attributes, n_classes
        self.diffusion_steps, self.steps = diffusion_steps, steps
        self.batch_size, self.learning_rate, self.seed, self.device = batch_size, learning_rate, seed, device
        torch.manual_seed(seed)
        betas = torch.linspace(1e-4, 0.02, diffusion_steps, dtype=torch.float64)
        alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(alphas, 0).float().to(device)
        self.betas = betas.float().to(device)
        self.alphas = alphas.float().to(device)
        time_embedding_size, class_embedding_size = 128, 64
        layers, width = [], n_attributes + time_embedding_size + class_embedding_size
        for hidden_width in hidden_layers:
            layers += [nn.Linear(width, hidden_width), nn.SiLU()]
            width = hidden_width
        layers += [nn.Linear(width, n_attributes)]
        self.denoiser = nn.Sequential(*layers).to(device)
        self.class_embedding = nn.Embedding(n_classes, class_embedding_size).to(device)
        self.time_embedding_size = time_embedding_size

    def _time_embedding(self, t):
        import torch
        half = self.time_embedding_size // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=self.device).float() / half)
        angles = t.float()[:, None] * frequencies[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], 1)

    def _predict_noise(self, x, t, labels):
        import torch
        return self.denoiser(torch.cat([x, self._time_embedding(t), self.class_embedding(labels)], 1))

    def fit(self, Z: np.ndarray, labels: np.ndarray, verbose=False):
        """Train the denoiser for `steps` mini-batches of rows, each noised to a random step t."""
        import torch
        X = torch.tensor(Z, dtype=torch.float32, device=self.device)
        C = torch.tensor(labels, dtype=torch.long, device=self.device)
        parameters = list(self.denoiser.parameters()) + list(self.class_embedding.parameters())
        optimiser = torch.optim.AdamW(parameters, lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, self.steps)
        n = len(X)
        batch_size = min(self.batch_size, n)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        for step in range(self.steps):
            rows = torch.randint(0, n, (batch_size,), generator=generator).to(self.device)
            x0, batch_labels = X[rows], C[rows]
            t = torch.randint(0, self.diffusion_steps, (batch_size,), generator=generator).to(self.device)
            noise = torch.randn(x0.shape, generator=generator).to(self.device)
            alpha_bar = self.alpha_bars[t][:, None]
            xt = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise
            loss = ((self._predict_noise(xt, t, batch_labels) - noise) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
            if verbose and (step % 2000 == 0 or step == self.steps - 1):
                print(f"      diffusion model step {step}: loss {float(loss):.4f}", flush=True)
        return self

    def sample(self, labels: np.ndarray, seed=0) -> np.ndarray:
        """One generated row per entry of `labels` (the class of each row)."""
        import torch
        with torch.no_grad():
            generator = torch.Generator(device="cpu").manual_seed(seed)
            n = len(labels)
            row_labels = torch.tensor(labels, dtype=torch.long, device=self.device)
            x = torch.randn((n, self.n_attributes), generator=generator).to(self.device)
            for t in reversed(range(self.diffusion_steps)):
                t_batch = torch.full((n,), t, device=self.device, dtype=torch.long)
                predicted_noise = self._predict_noise(x, t_batch, row_labels)
                alpha, alpha_bar, beta = self.alphas[t], self.alpha_bars[t], self.betas[t]
                mean = (x - (beta / (1 - alpha_bar).sqrt()) * predicted_noise) / alpha.sqrt()
                if t > 0:
                    x = mean + beta.sqrt() * torch.randn((n, self.n_attributes), generator=generator).to(self.device)
                else:
                    x = mean
            return x.cpu().numpy().astype(np.float64)


class TabDDPMFailureGenerator(FailureRowGenerator):
    """The diffusion model above, fitted on the failure rows and conditioned on the cluster label. `steps` is the
    number of training mini-batches (20,000 in the arm); it runs on a GPU when one is available."""

    def __init__(self, steps=20000, hidden_layers=(512, 512, 512), diffusion_steps=1000, batch_size=256,
                 learning_rate=1e-3, device=None, **kwargs):
        super().__init__(**kwargs)
        self.steps, self.hidden_layers, self.diffusion_steps = steps, hidden_layers, diffusion_steps
        self.batch_size, self.learning_rate = batch_size, learning_rate
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_failures):
        started = time.time()
        Z = self._prepare(X_failures)
        self.model_ = GaussianDDPM(Z.shape[1], n_classes=self.n_clusters, hidden_layers=tuple(self.hidden_layers),
                                   diffusion_steps=self.diffusion_steps, steps=self.steps, batch_size=self.batch_size,
                                   learning_rate=self.learning_rate, seed=self.seed, device=self.device)
        self.model_.fit(Z, self.cluster_labels_, verbose=self.verbose)
        if self.verbose:
            print(f"    TabDDPM fit ({self.steps} steps, {self.diffusion_steps} diffusion steps) on {Z.shape} in "
                  f"{time.time() - started:.0f}s [{self.device}]", flush=True)
        return self

    def sample(self, n, reapply_missingness=True, seed=None):
        seed = self.seed if seed is None else seed
        rng = np.random.default_rng(seed)
        started = time.time()
        Z = self.model_.sample(self._draw_cluster_labels(n, rng), seed=seed)
        Z = np.clip(Z, -6, 6)          # standardised space: guards against the odd diverging sampling chain
        if self.verbose:
            print(f"    TabDDPM sampled {n} rows in {time.time() - started:.0f}s", flush=True)
        return self._finish(self._inverse(Z), reapply_missingness, seed)


# ---------------------------------------------------------------------------------------------------------- CTAB-GAN+
CTAB_GAN_PLUS_REPOSITORY = Path(os.environ.get("APS_CTAB_GAN_PLUS_REPOSITORY", CTAB_GAN_PLUS_REPOSITORY_DIR))
CTAB_GAN_PLUS_COMMIT = "6a6f90188cca3dac2c533fd5e8e7f20de074365b"   # the commit used for the thesis


def add_ctab_gan_plus_repository_to_path():
    """Make the CTAB-GAN+ repository importable (its code is imported as the package `model`)."""
    if not (CTAB_GAN_PLUS_REPOSITORY / "model" / "ctabgan.py").exists():
        raise FileNotFoundError(
            f"The CTAB-GAN+ repository was not found at {CTAB_GAN_PLUS_REPOSITORY}. Run "
            f"`bash third_party/fetch_third_party_code.sh` from the repository root, which clones "
            f"https://github.com/Team-TUD/CTAB-GAN-Plus at commit {CTAB_GAN_PLUS_COMMIT}, or set "
            f"APS_CTAB_GAN_PLUS_REPOSITORY to an existing clone of that commit.")
    if str(CTAB_GAN_PLUS_REPOSITORY) not in sys.path:
        sys.path.insert(0, str(CTAB_GAN_PLUS_REPOSITORY))


class CTABGANPlusFailureGenerator(FailureRowGenerator):
    """CTAB-GAN+ fitted on the failure rows, with the authors' code from third_party/CTAB-GAN-Plus.

    All attributes are passed as plain continuous columns in the standardised log space, so the repository's own
    log transform for long-tailed columns is not used; the cluster label is both the categorical column of the
    conditional vector and the "Classification" target of the auxiliary classifier. The repository reads its
    training table from a CSV file, which is written to a temporary folder and deleted after sampling.
    """

    def __init__(self, epochs=150, batch_size=200, **kwargs):
        super().__init__(**kwargs)
        self.epochs, self.batch_size = epochs, batch_size

    def fit(self, X_failures):
        add_ctab_gan_plus_repository_to_path()
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        from model.ctabgan import CTABGAN
        started = time.time()
        Z = self._prepare(X_failures)
        table = pd.DataFrame(Z, columns=[f"f{i}" for i in range(Z.shape[1])])
        table["aux_cluster"] = self.cluster_labels_.astype(int)
        self.temporary_dir_ = Path(tempfile.mkdtemp(prefix="ctabgan_"))
        csv_file = self.temporary_dir_ / "pos.csv"
        table.to_csv(csv_file, index=False)
        # DataPrep insists on a stratified hold-out of its own; keep it to three rows (its smallest valid size)
        self.model_ = CTABGAN(raw_csv_path=str(csv_file), test_ratio=min(0.5, 3.0 / len(table)),
                              categorical_columns=["aux_cluster"], log_columns=[], mixed_columns={}, general_columns=[],
                              non_categorical_columns=[], integer_columns=[],
                              problem_type={"Classification": "aux_cluster"})
        self.model_.synthesizer.epochs = int(self.epochs)
        # the batch size is rounded down to a multiple of 10 (at least 10), the same rule as for CTGAN above
        self.model_.synthesizer.batch_size = max(10, (min(self.batch_size, len(table)) // 10) * 10)
        with contextlib.redirect_stdout(io.StringIO()):       # the repository prints a progress bar per epoch
            self.model_.fit()
        if self.verbose:
            print(f"    CTAB-GAN+ fit ({self.epochs} epochs) on {Z.shape} in {time.time() - started:.0f}s", flush=True)
        return self

    def _generate_clipped(self, n: int) -> np.ndarray:
        """Draw n rows from the fitted generator, clipping out-of-range values instead of rejecting rows.

        The repository's own sample() rejects every generated row that has any attribute outside the training range
        and draws again until n rows survive; with 160 continuous attributes that loop does not terminate. This
        version keeps the generator, the conditional vector and the inverse transform untouched, but clips an
        out-of-range value to the range instead of discarding the row. That is the only deviation from the
        original code.
        """
        import torch
        from model.synthesizer.ctabgan_synthesizer import apply_activate
        synthesizer = self.model_.synthesizer
        synthesizer.generator.eval()
        transformer = synthesizer.transformer
        output_info = transformer.output_info
        batches = n // synthesizer.batch_size + 1
        generated = []
        with torch.no_grad():
            for _ in range(batches):
                noise = torch.randn(synthesizer.batch_size, synthesizer.random_dim, device=synthesizer.device)
                condition = torch.from_numpy(synthesizer.cond_generator.sample(synthesizer.batch_size))
                condition = condition.to(synthesizer.device)
                noise = torch.cat([noise, condition], dim=1).view(
                    synthesizer.batch_size, synthesizer.random_dim + synthesizer.cond_generator.n_opt, 1, 1)
                fake = synthesizer.generator(noise)
                fake_rows = synthesizer.Gtransformer.inverse_transform(fake)
                generated.append(apply_activate(fake_rows, output_info).detach().cpu().numpy())
        generated = np.concatenate(generated, axis=0)[:n]
        saved_ranges = [(meta.get("min"), meta.get("max")) for meta in transformer.meta]
        for meta in transformer.meta:                       # switch the range rejection off for this call
            if meta["type"] in ("continuous", "mixed"):
                meta["min"], meta["max"] = -np.inf, np.inf
        try:
            rows, _ = transformer.inverse_transform(generated)
        finally:
            for meta, (low, high) in zip(transformer.meta, saved_ranges):
                if low is not None:
                    meta["min"], meta["max"] = low, high
        for j, meta in enumerate(transformer.meta):         # clip instead of reject
            if meta["type"] == "continuous" and meta.get("min") is not None:
                rows[:, j] = np.clip(rows[:, j].astype(np.float64), meta["min"], meta["max"])
        return rows

    def sample(self, n, reapply_missingness=True, seed=None):
        import torch
        seed = self.seed if seed is None else seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        raw_rows = self._generate_clipped(n)
        rows = self.model_.data_prep.inverse_prep(raw_rows)
        rows = rows.drop(columns=["aux_cluster"]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        Z = rows.to_numpy(dtype=np.float64)
        shutil.rmtree(self.temporary_dir_, ignore_errors=True)
        return self._finish(self._inverse(Z), reapply_missingness, seed)


# -------------------------------------------------------------------------------------------------------------- GReaT
def _disable_old_torchao():
    """Let peft build LoRA layers next to an old torchao.

    peft refuses to build LoRA layers when a torchao older than 0.16 is installed, even though nothing here uses
    torchao; the Colab runtime of the thesis runs ships torchao 0.10 with its PyTorch, and the first run of this arm
    stopped with an ImportError in get_peft_model. The LoRA dispatcher only asks whether torchao is available;
    answering "no" makes it skip the torchao code path (torchao serves quantised weights), which is the right path for
    the unquantised weights, in bfloat16 or 32-bit floating point, used in this arm.
    """
    try:
        import importlib.metadata as metadata
        from packaging import version
        if version.parse(metadata.version("torchao")) >= version.parse("0.16.0"):
            return
    except Exception:  # noqa: BLE001  (torchao absent or unreadable: nothing to do)
        return
    importlib.import_module("peft.tuners.lora")   # loads peft and the LoRA dispatcher modules that bind the name
    for name, module in list(sys.modules.items()):
        if name.startswith("peft") and getattr(module, "is_torchao_available", None) is not None:
            module.is_torchao_available = lambda: False
    print("    note: old torchao found; peft's torchao integration disabled for this process", flush=True)


class GReaTFailureGenerator(FailureRowGenerator):
    """GReaT (Borisov et al., ICLR 2023): failure rows as sentences for a language model.

    Every failure row is written as a short text, "f3 is 1234, f7 is 0, ..., aux_cluster is c1": the attributes in
    raw counter units rounded to integers (no log map, no standardisation; the k-means clusters are therefore also
    computed in raw units), with the column names shortened to f<i> to save tokens. A pretrained causal language
    model (`llm`) is fine-tuned on these sentences with LoRA adapters: the pretrained weights stay frozen and small
    low-rank matrices added to the attention projections are trained instead (rank lora_r). New rows are generated
    as text from a prompt that fixes the cluster label, so sampling is conditional on the cluster, and the text is
    parsed back into numbers. Needs a GPU for anything larger than a toy model.

    On APS this arm failed. One serialised failure row is about 2,200 tokens long (the Qwen tokenizer writes every
    digit as a token of its own), but the model's generation configuration capped every sequence at 2,048 new
    tokens, a setting that takes precedence over a `max_length` argument. Every sampled row was therefore cut off
    before its last attributes, and no row could be parsed. The code here was corrected afterwards: `sample` now
    passes the limit as max_new_tokens (3,000 by default), which that configuration does not override. The corrected
    version has not been run on APS; one fold of the failed run took about seven and a half hours on an L4 (an NVIDIA
    GPU with 24 GB of memory), and the remaining compute was spent elsewhere.

    With bf16 (the default), fine-tuning on a GPU uses bfloat16 mixed precision: most arithmetic is done in bfloat16,
    a 16-bit floating-point format with the exponent range of 32-bit floats. max_attributes_as_text is for smoke tests
    only: just the first attributes are written as text, and the others come back as the real medians.
    """

    def __init__(self, llm="Qwen/Qwen2.5-3B", epochs=10, batch_size=8, lora_r=8, max_new_tokens=3000, rows_per_pass=50,
                 temperature=0.7, bf16=True, max_attributes_as_text=None, min_parsed_fraction=0.95, **kwargs):
        super().__init__(log_space=False, standardise=False, **kwargs)
        self.min_parsed_fraction = min_parsed_fraction
        self.llm, self.epochs, self.batch_size, self.lora_r = llm, epochs, batch_size, lora_r
        self.max_new_tokens, self.rows_per_pass = max_new_tokens, rows_per_pass
        self.temperature, self.bf16 = temperature, bf16
        self.max_attributes_as_text = max_attributes_as_text

    def fit(self, X_failures):
        from be_great import GReaT
        import torch
        started = time.time()
        values = self._prepare(X_failures)            # raw units (shift only), constant attributes left out
        table = pd.DataFrame(np.round(values).astype(np.int64), columns=[f"f{i}" for i in range(values.shape[1])])
        self.fill_values_ = np.median(values, axis=0)  # used for attributes that cannot be parsed or are not written
        if self.max_attributes_as_text:
            table = table.iloc[:, :int(self.max_attributes_as_text)]
        table["aux_cluster"] = [f"c{cluster}" for cluster in self.cluster_labels_]
        self.text_columns_ = list(table.columns)
        experiment_dir = tempfile.mkdtemp(prefix="great_")
        # LoRA target modules: attention projections of Llama/Qwen-style models, the fused c_attn of GPT-2-style ones
        from transformers import AutoConfig
        architectures = AutoConfig.from_pretrained(self.llm).architectures or [""]
        targets = ["c_attn"] if "gpt2" in str(architectures[0]).lower() else ["q_proj", "v_proj"]
        great_options = dict(epochs=self.epochs, batch_size=self.batch_size, efficient_finetuning="lora",
                             lora_config={"r": self.lora_r, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none",
                                          "target_modules": targets, "task_type": "CAUSAL_LM"},
                             save_strategy="no", logging_steps=50, report_to=[], seed=self.seed)
        if self.bf16 and torch.cuda.is_available():
            great_options["bf16"] = True
        _disable_old_torchao()
        self.model_ = GReaT(self.llm, experiment_dir=experiment_dir, **great_options)
        self.model_.fit(table, conditional_col="aux_cluster")
        self.experiment_dir_ = experiment_dir
        if self.verbose:
            print(f"    GReaT fit ({self.llm}, {self.epochs} epochs, LoRA r={self.lora_r}) on {values.shape} in "
                  f"{time.time() - started:.0f}s", flush=True)
        return self

    def sample(self, n, reapply_missingness=True, seed=None, max_passes=None, time_budget_seconds=None,
               min_parsed_fraction=None):
        """Bounded sampler.

        be_great's own loop runs until n valid rows exist, which can take indefinitely long when the model writes a
        few malformed attributes in every row. Here each pass generates rows_per_pass rows from the cluster prompt
        and keeps those in which at least min_parsed_fraction of the attributes parse as numbers (the rest are
        filled with the real medians). Sampling stops at n rows, after max_passes passes or when the time budget
        (APS_GREAT_TIME_BUDGET_SECONDS, default three hours) is used up, whichever comes first. If fewer than n rows
        were kept, the kept rows are drawn again with replacement up to n (sample_stats_ records how many); if none
        was kept, the run stops with an error.
        """
        import torch
        from be_great.great_utils import _convert_tokens_to_text, _convert_text_to_tabular_data
        seed = self.seed if seed is None else seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        started = time.time()
        max_passes = max_passes or max(10, 4 * n // self.rows_per_pass + 10)
        min_parsed_fraction = self.min_parsed_fraction if min_parsed_fraction is None else min_parsed_fraction
        time_budget_seconds = time_budget_seconds or float(os.environ.get("APS_GREAT_TIME_BUDGET_SECONDS", 3 * 3600))
        cluster_distribution = {f"c{cluster}": float(share) for cluster, share in enumerate(self.cluster_proportions_)}
        great = self.model_
        great._resolve_device("cuda" if torch.cuda.is_available() else "cpu")
        start_sampler = great._get_start_sampler("aux_cluster", cluster_distribution)
        attribute_columns = [column for column in self.text_columns_ if column != "aux_cluster"]
        medians = pd.Series(self.fill_values_[:len(attribute_columns)], index=attribute_columns)
        kept, rows_seen, passes = [], 0, 0
        while (sum(len(part) for part in kept) < n and passes < max_passes
               and time.time() - started < time_budget_seconds):
            passes += 1
            start_tokens = torch.tensor(start_sampler.get_start_tokens(self.rows_per_pass)).to(great.device)
            attention_mask = (start_tokens != great.tokenizer.pad_token_id).long()
            with torch.no_grad():
                # max_new_tokens rather than max_length: see the class docstring
                tokens = great.model.generate(input_ids=start_tokens, attention_mask=attention_mask,
                                              max_new_tokens=self.max_new_tokens, do_sample=True,
                                              temperature=self.temperature, pad_token_id=great.tokenizer.pad_token_id)
            rows = _convert_text_to_tabular_data(_convert_tokens_to_text(tokens, great.tokenizer), great.columns)
            rows = rows.replace("placeholder", np.nan)
            rows = rows[attribute_columns].apply(pd.to_numeric, errors="coerce")
            parsed_fraction = rows.notna().mean(axis=1)
            rows_seen += len(rows)
            accepted = rows[parsed_fraction >= min_parsed_fraction].fillna(medians)
            kept.append(accepted)
            if self.verbose and (passes <= 3 or passes % 10 == 0):
                print(f"      GReaT pass {passes}: {len(accepted)}/{len(rows)} rows kept (median parseable fraction "
                      f"{float(parsed_fraction.median()):.2f}); {sum(len(part) for part in kept)}/{n} so far, "
                      f"{time.time() - started:.0f}s", flush=True)
        generated = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=attribute_columns)
        self.sample_stats_ = {"requested": n, "generated": int(len(generated)), "passes": passes,
                              "rows_seen": rows_seen, "seconds": time.time() - started}
        if len(generated) < n:          # not enough valid rows inside the budget: draw the valid ones again
            if len(generated) == 0:
                raise RuntimeError("GReaT produced no parseable rows")
            repeated = np.random.default_rng(seed).integers(0, len(generated), n - len(generated))
            generated = pd.concat([generated, generated.iloc[repeated]], ignore_index=True)
            self.sample_stats_["duplicated_to_fill"] = int(len(repeated))
        values = generated.iloc[:n].to_numpy(dtype=np.float64)
        if values.shape[1] < len(self.fill_values_):  # smoke-test cap on the attributes written as text
            values = np.hstack([values, np.tile(self.fill_values_[values.shape[1]:], (len(values), 1))])
        if self.verbose:
            print(f"    GReaT sample: {self.sample_stats_}", flush=True)
        shutil.rmtree(self.experiment_dir_, ignore_errors=True)
        return self._finish(self._inverse(values), reapply_missingness, seed)


# ---------------------------------------------------------------------------------------------------------- augmenter
SURVEY_GENERATORS = {"gaussian_copula": GaussianCopulaFailureGenerator, "tvae": TVAEFailureGenerator,
                     "ctgan": CTGANFailureGenerator, "tabddpm": TabDDPMFailureGenerator,
                     "ctab_gan_plus": CTABGANPlusFailureGenerator, "great": GReaTFailureGenerator}


def survey_generator_augmenter(kind: str, n_synthetic=2000, reapply_missingness=True, **generator_options):
    """Augmenter for run_cross_validation: fits SURVEY_GENERATORS[kind] (with generator_options passed to its
    constructor) on the training fold's failure rows with the fold's seed, and appends n_synthetic rows sampled with
    seed + 1."""
    generator_class = SURVEY_GENERATORS[kind]

    def augment(X_train: pd.DataFrame, y_train: np.ndarray, seed: int):
        generator = generator_class(seed=seed, **generator_options).fit(X_train[y_train == 1])
        X_synthetic = generator.sample(n_synthetic, reapply_missingness=reapply_missingness, seed=seed + 1)
        X_augmented = pd.concat([X_train, X_synthetic], ignore_index=True)
        y_augmented = np.concatenate([y_train, np.ones(len(X_synthetic), dtype=y_train.dtype)])
        return X_augmented, y_augmented
    return augment
