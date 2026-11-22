"""A PyTorch port of the cost-sensitive Transformer of Beikmohammadi et al. (RQ1b).

Beikmohammadi et al. (Cluster Computing, 2026) classify the APS data with a small Transformer written in Keras
(TrainingCode.py of their public repository, commit 4db444a). This module re-implements it layer by layer, so that it
can be trained under the published protocol and the protocol with a validation split of real rows
(run_transformer_2026_port.py) and with ablations. The original model, in Keras terms:

    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
        x = LayerNormalization(eps=1e-6)(inputs)
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        res = x + inputs
        x = LayerNormalization(eps=1e-6)(res)
        x = Conv1D(ff_dim, 1, relu)(x)
        x = Dropout(dropout)(x)
        x = Conv1D(C, 1)(x)
        return x + res

    inputs (batch, 142, 1) -> 4 encoder blocks -> GlobalAveragePooling1D(channels_first)
    -> Dense(128, relu) -> Dropout(0.4) -> Dense(64, relu) -> Dropout(0.4) -> Dense(1, sigmoid)

Each scaled attribute is one token with a single channel, so self-attention relates the attributes of one truck to
each other. Pooling "channels_first" over a single channel simply flattens the tokens into a vector of 142 values.
The settings are head_size 256, 4 heads, feed-forward width 4, dropout 0.25 in the blocks and 0.4 in the classifier
head, Adam with learning rate 5e-4, batches of 72 rows and 8,000 epochs. With 142 attributes the port has 55,369
trainable parameters, the same number as the authors' saved models; the runs of this repository have 141 attributes,
because the constant attribute cd_000 is dropped first.

The loss is the binary focal loss of the focal_loss package, BinaryFocalLoss(pos_weight=0.95, gamma=1.5). The focal
loss multiplies the cross-entropy of each row by (1 - p_t)^gamma, where p_t is the predicted probability of the
row's true class, so rows that are already classified confidently contribute little and training concentrates on
the hard ones. In the package, pos_weight multiplies the term of the positive rows only:

    L = -pos_weight * y * (1 - p)^gamma * log(p) - (1 - y) * p^gamma * log(1 - p)

With pos_weight 0.95 the trained loss is therefore close to symmetric between the classes, not the alpha / (1 -
alpha) weighting of failures against healthy trucks written in the paper's equations. Both are available:
loss_mode="package" (the default, what the authors' code trains) and loss_mode="paper".

Initialisation. The authors' model sets no initialisers of its own, so Keras' defaults apply: Glorot-uniform kernels
and zero biases. PyTorch's defaults differ (Kaiming-uniform kernels and random biases), and the port also uses
Keras' Adam epsilon of 1e-7 instead of PyTorch's 1e-8 (epsilon is added to the denominator of every Adam step, so it
matters when gradients are small). Matching both removes two differences between the port and the original that
have nothing to do with the method. initialise_like_keras gives every kernel the fan-in and fan-out that Keras
computes for the kernel's shape in the original layer.

Checkpoints. Long runs on Colab are interrupted, so train_transformer can save the full training state (weights,
optimiser, history, the best weights so far) every few epochs and resume from it when the same command is issued
again. A checkpoint trained with a different initialisation is set aside rather than resumed. A run whose
validation predictions have been constant for a given number of epochs (every truck predicted healthy, or every
truck predicted failing) has collapsed and is stopped.

The module has no command line; see run_transformer_2026_port.py and evaluate_port_checkpoint.py.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.common.data_loading import COST_OF_FALSE_ALARM, COST_OF_MISSED_FAILURE


class EncoderBlock(nn.Module):
    """One pre-normalised encoder block: layer norm, multi-head self-attention, residual; layer norm, two 1 x 1
    convolutions with ReLU and dropout in between, residual. Input and output have shape (batch, tokens, d_model).

    The submodule names (ln1, q, k, v, o, ln2, ff1, ff2) are the keys of the saved state dicts. They are kept as they
    are so that the checkpoints of the reported runs remain loadable.
    """

    def __init__(self, d_model, n_heads, head_size, ff_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.q = nn.Linear(d_model, n_heads * head_size)
        self.k = nn.Linear(d_model, n_heads * head_size)
        self.v = nn.Linear(d_model, n_heads * head_size)
        self.o = nn.Linear(n_heads * head_size, d_model)
        self.n_heads, self.head_size, self.attention_dropout = n_heads, head_size, dropout
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ff1 = nn.Conv1d(d_model, ff_dim, 1)
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.ff2 = nn.Conv1d(ff_dim, d_model, 1)

    def forward(self, x):
        batch, tokens, _ = x.shape
        h = self.ln1(x)
        query = self.q(h).view(batch, tokens, self.n_heads, self.head_size).transpose(1, 2)
        key = self.k(h).view(batch, tokens, self.n_heads, self.head_size).transpose(1, 2)
        value = self.v(h).view(batch, tokens, self.n_heads, self.head_size).transpose(1, 2)
        # Keras applies the dropout to the attention weights, as scaled_dot_product_attention does
        attended = F.scaled_dot_product_attention(query, key, value,
                                                  dropout_p=self.attention_dropout if self.training else 0.0)
        attended = self.o(attended.transpose(1, 2).reshape(batch, tokens, self.n_heads * self.head_size))
        residual = attended + x
        h = self.ln2(residual).transpose(1, 2)
        h = self.ff2(self.feed_forward_dropout(F.relu(self.ff1(h)))).transpose(1, 2)
        return h + residual


class Transformer2026Port(nn.Module):
    """The authors' network: n_blocks encoder blocks over n_features one-channel tokens, then an MLP head.

    forward takes a (batch, n_features) array of scaled attributes and returns one logit per row (the sigmoid of
    the Keras output layer is applied by the loss and by predict_proba).
    """

    def __init__(self, n_features, head_size=256, n_heads=4, ff_dim=4, n_blocks=4,
                 mlp_units=(128, 64), dropout=0.25, mlp_dropout=0.4):
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(1, n_heads, head_size, ff_dim, dropout) for _ in range(n_blocks)])
        head_layers, width = [], n_features
        for units in mlp_units:
            head_layers += [nn.Linear(width, units), nn.ReLU(), nn.Dropout(mlp_dropout)]
            width = units
        head_layers += [nn.Linear(width, 1)]
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        h = x.unsqueeze(-1)            # (batch, n_features) -> (batch, n_features tokens, 1 channel)
        for block in self.blocks:
            h = block(h)
        return self.head(h.squeeze(-1)).squeeze(-1)


def _glorot_uniform_(weight, fan_in, fan_out):
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        weight.uniform_(-limit, limit)


def initialise_like_keras(model):
    """Initialise the port as Keras initialises the authors' model: Glorot-uniform kernels and zero biases.

    Glorot-uniform draws from U(-a, a) with a = sqrt(6 / (fan_in + fan_out)), and Keras derives the fans from the
    shape of each kernel: (d, heads, head_size) for the query, key and value kernels of MultiHeadAttention, (heads,
    head_size, d) for its output kernel, (1, in, out) for Conv1D and (in, out) for Dense. The fans below are the ones
    Keras gets for those shapes. Layer norms start at weight 1 and bias 0 in both libraries.
    """
    for module in model.modules():
        if isinstance(module, EncoderBlock):
            d, heads, head_size = module.q.in_features, module.n_heads, module.head_size
            for projection in (module.q, module.k, module.v):
                _glorot_uniform_(projection.weight, heads * d, head_size * d)
                nn.init.zeros_(projection.bias)
            _glorot_uniform_(module.o.weight, head_size * heads, d * heads)
            nn.init.zeros_(module.o.bias)
            for convolution in (module.ff1, module.ff2):
                _glorot_uniform_(convolution.weight, convolution.in_channels, convolution.out_channels)
                nn.init.zeros_(convolution.bias)
    for layer in model.head:
        if isinstance(layer, nn.Linear):
            _glorot_uniform_(layer.weight, layer.in_features, layer.out_features)
            nn.init.zeros_(layer.bias)
    return model


def _save_checkpoint(state, path, verbose=True):
    """Checkpoint save that never ends a run: a failed write (e.g. a Google Drive hiccup) is retried once and then
    skipped; training continues and the next scheduled save tries again."""
    for attempt in (1, 2):
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save(state, path)
            return True
        except Exception as error:  # noqa: BLE001
            if verbose:
                print(f"    checkpoint save failed ({type(error).__name__}: {error}); attempt {attempt}", flush=True)
            time.sleep(10)
    return False


def binary_focal_loss(logits, y, pos_weight=0.95, gamma=1.5, loss_mode="package"):
    """Mean focal loss of a batch (see the module docstring).

    loss_mode="package": focal_loss.BinaryFocalLoss, pos_weight on the positive term only.
    loss_mode="paper":   pos_weight on the positive term and (1 - pos_weight) on the negative term.
    """
    probability = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    y = y.float()
    positive_term = -(1 - probability) ** gamma * torch.log(probability)
    negative_term = -(probability ** gamma) * torch.log(1 - probability)
    positive_weight = pos_weight
    negative_weight = 1.0 if loss_mode == "package" else (1.0 - pos_weight)
    return (positive_weight * y * positive_term + negative_weight * (1 - y) * negative_term).mean()


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _cost_fn_fp(y, decisions):
    fn = int(((y == 1) & (decisions == 0)).sum())
    fp = int(((y == 0) & (decisions == 1)).sum())
    return COST_OF_MISSED_FAILURE * fn + COST_OF_FALSE_ALARM * fp, fn, fp


def train_transformer(X_train, y_train, X_val, y_val, *, epochs=300, patience=250, batch_size=72, learning_rate=5e-4,
                      pos_weight=0.95, gamma=1.5, loss_mode="package", architecture: dict | None = None, seed=0,
                      device="cpu", threshold=0.5, log_every=1, max_minutes=None, verbose=True, amp=False,
                      checkpoint_path=None, checkpoint_every=100, collapse_patience=None, initialisation="keras",
                      adam_epsilon=1e-7):
    """Train a Transformer2026Port and keep the weights of the best validation epoch.

    The best epoch is the one with the lowest validation cost at `threshold` (0.5, the published cut-off), which is
    the checkpoint rule of the authors' code (ModelCheckpoint on the validation cost). Training stops after `epochs`
    epochs, after `patience` epochs without a new best, or after max_minutes. The weights of the last epoch are
    returned as well, because the authors' code scores the final-epoch model.

    architecture: keyword arguments of Transformer2026Port. amp: mixed precision on a GPU.
    checkpoint_path, checkpoint_every: save the full training state to this file every checkpoint_every epochs (and
        at the end) and resume from it if it exists when training starts.
    collapse_patience: stop when the validation predictions have been constant for this many consecutive epochs.
        The best checkpoint is kept, the stop is recorded in the history and the checkpoint, and a resumed run (for
        example an extension to more epochs) that starts from a stopped checkpoint does not train further.
    initialisation: "keras" (initialise_like_keras) or "torch" (PyTorch's defaults); adam_epsilon: Adam's epsilon.

    Returns (model with the best weights loaded, history (one dict per epoch), validation probabilities at the best
    epoch, best epoch, state dict of the last epoch).
    """
    # Model and optimiser, built and initialised from the seed.
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Transformer2026Port(X_train.shape[1], **(architecture or {}))
    if initialisation == "keras":
        initialise_like_keras(model)
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=adam_epsilon)
    # Resuming: the checkpoint of an interrupted run, if there is one, replaces the weights and the optimiser state.
    start_epoch = 0
    resumed = None
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        try:
            resumed = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as error:  # noqa: BLE001  (a checkpoint cut off mid-write: start afresh, not fail every time)
            print(f"    checkpoint {checkpoint_path} unreadable ({type(error).__name__}); starting afresh", flush=True)
            resumed = None
        # the checkpoint's "init" key records its initialisation; files written before the key existed used PyTorch's
        if resumed is not None and resumed.get("init", "torch") != initialisation:
            print(f"    checkpoint was trained with init={resumed.get('init', 'torch')}, this run uses "
                  f"init={initialisation}: set aside as {checkpoint_path}.{resumed.get('init', 'torch')}init.bak; "
                  f"starting afresh", flush=True)
            try:
                os.replace(checkpoint_path, f"{checkpoint_path}.{resumed.get('init', 'torch')}init.bak")
            except Exception:  # noqa: BLE001
                pass
            resumed = None
        if resumed is not None:
            model.load_state_dict(resumed["model"])
            optimiser.load_state_dict(resumed["opt"])
            start_epoch = resumed["epoch"] + 1
            if verbose:
                print(f"    resumed from {checkpoint_path} at epoch {start_epoch}", flush=True)
    # Inputs as tensors on the training device.
    X_train_tensor = torch.tensor(np.asarray(X_train, dtype=np.float32), device=device)
    y_train_tensor = torch.tensor(np.asarray(y_train, dtype=np.int64), device=device)
    X_val_tensor = torch.tensor(np.asarray(X_val, dtype=np.float32), device=device)
    y_val = np.asarray(y_val, dtype=np.int64)
    n_train = len(y_train_tensor)
    n_val_failures = int(y_val.sum())
    n_val_healthy = len(y_val) - n_val_failures
    # constant predictions: every validation truck predicted healthy, or every one predicted failing
    predictions_constant = lambda fn, fp: (fp == 0 and fn == n_val_failures) or (fn == 0 and fp == n_val_healthy)
    gradient_scaler = torch.amp.GradScaler("cuda", enabled=(amp and device == "cuda"))
    # The best epoch so far as (validation cost, state dict, validation probabilities, epoch), and the bookkeeping
    # of the stopping rules; a resumed run takes all of it over from the checkpoint.
    best = (math.inf, None, None, -1)
    epochs_without_improvement = 0
    history = []
    started = time.time()
    if resumed is not None:
        best = (resumed["best_cost"], resumed["best_state"], resumed["best_pv"], resumed["best_epoch"])
        epochs_without_improvement = resumed["bad"]
        history = resumed["hist"]
        started -= history[-1]["seconds"] if history else 0
    constant_epochs = 0                       # current run of consecutive constant-prediction epochs
    for entry in reversed(history):
        if not predictions_constant(entry["val_fn"], entry["val_fp"]):
            break
        constant_epochs += 1
    stopped = resumed.get("stopped") if resumed is not None else None
    if stopped and verbose:
        print(f"    checkpoint was stopped ({stopped}); not training further", flush=True)
    if verbose:
        print(f"    parameters {count_trainable_parameters(model):,}; training rows {n_train}; "
              f"validation rows {len(y_val)}; device {device}", flush=True)
    # Training loop: one pass over the training rows in random order, then the validation cost at `threshold`.
    for epoch in (range(start_epoch, epochs) if not stopped else range(0)):
        model.train()
        permutation = torch.randperm(n_train, device=device)
        loss_sum = 0.0
        for first in range(0, n_train, batch_size):
            batch_index = permutation[first:first + batch_size]
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(amp and device == "cuda")):
                loss = binary_focal_loss(model(X_train_tensor[batch_index]), y_train_tensor[batch_index], pos_weight,
                                         gamma, loss_mode)
            gradient_scaler.scale(loss).backward()
            gradient_scaler.step(optimiser)
            gradient_scaler.update()
            loss_sum += float(loss.detach()) * len(batch_index)
        model.eval()
        probability_batches = []
        with torch.no_grad():
            for first in range(0, len(X_val_tensor), 256):
                probability_batches.append(torch.sigmoid(model(X_val_tensor[first:first + 256])).float().cpu().numpy())
        val_probabilities = np.concatenate(probability_batches)
        cost, fn, fp = _cost_fn_fp(y_val, (val_probabilities >= threshold).astype(int))
        constant_epochs = constant_epochs + 1 if predictions_constant(fn, fp) else 0
        history.append({"epoch": epoch, "train_loss": loss_sum / n_train, "val_cost": cost, "val_fn": fn, "val_fp": fp,
                        "seconds": time.time() - started})
        collapsed = collapse_patience is not None and constant_epochs >= collapse_patience
        if collapsed:
            stopped = "collapsed"
            history[-1]["stop"] = f"collapsed: constant predictions since epoch {epoch - constant_epochs + 1}"
            if verbose:
                print(f"    stopped at epoch {epoch}: constant predictions for {constant_epochs} epochs (since epoch "
                      f"{epoch - constant_epochs + 1}); best epoch {best[3] if cost >= best[0] else epoch}", flush=True)
        if verbose and (epoch % log_every == 0):
            print(f"    epoch {epoch:4d} loss {loss_sum / n_train:.5f} validation cost {cost} (FN {fn} FP {fp}) "
                  f"{time.time() - started:.0f}s", flush=True)
        # best epoch so far (the authors' ModelCheckpoint rule), checkpoint, and the stopping rules
        if cost < best[0]:
            best = (cost, {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
                    val_probabilities.copy(), epoch)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if checkpoint_path is not None and ((epoch + 1) % checkpoint_every == 0 or epoch == epochs - 1
                                            or epochs_without_improvement >= patience or collapsed):
            # The key names are the checkpoint format read by evaluate_port_checkpoint.py and by resumed runs:
            # model and opt (state dicts of the model and of the optimiser), epoch (the last finished epoch), hist (the
            # history), bad (epochs without a new best), best_cost, best_state, best_pv (the validation
            # probabilities) and best_epoch (the four parts of `best`), stopped (None or the reason for stopping)
            # and init (the initialisation).
            _save_checkpoint({"model": model.state_dict(), "opt": optimiser.state_dict(), "epoch": epoch,
                              "hist": history, "bad": epochs_without_improvement, "best_cost": best[0],
                              "best_state": best[1], "best_pv": best[2], "best_epoch": best[3], "stopped": stopped,
                              "init": initialisation}, checkpoint_path, verbose)
        if epochs_without_improvement >= patience or collapsed:
            break
        if max_minutes is not None and (time.time() - started) / 60 > max_minutes:
            if verbose:
                print(f"    time budget reached after epoch {epoch}", flush=True)
            break
    # Final weights of the last epoch, and the model returned with the weights of the best epoch.
    final_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    model.load_state_dict(best[1])
    return model, history, best[2], best[3], final_state


@torch.no_grad()
def predict_proba(model, X, device="cpu"):
    """Predicted failure probabilities (sigmoid of the logits) for the rows of X, in batches of 256."""
    model.eval()
    X_tensor = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    probability_batches = []
    for first in range(0, len(X_tensor), 256):
        probability_batches.append(torch.sigmoid(model(X_tensor[first:first + 256])).float().cpu().numpy())
    return np.concatenate(probability_batches)
