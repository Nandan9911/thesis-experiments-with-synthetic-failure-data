"""Time per training epoch of the cost-sensitive Transformer of Beikmohammadi et al. on the local CPU (RQ1b, a planning
measurement).

Before deciding where to retrain the authors' model, this script measured the step time of exactly their
architecture (the transformer_encoder and build_model functions of TrainingCode.py, copied below; 55,369 trainable
parameters with 142 attributes) on random data, for the published batch size of 72 and for 256 and 512. One epoch
over the published training split, 90 % of the 86,636 resampled rows (77,972 rows), is then extrapolated from the
time per step. On the two-core machine used for the CPU experiments it came to about 2,900 seconds per epoch at batch
size 72, so the 8,000 published epochs were out of reach without a GPU; the retraining runs were made on Colab GPUs
instead.

    python -m experiments.rq1_transformer_2026_reproduction.authors_code.time_one_training_epoch

Reads nothing (the data are random) and writes nothing; the console output of the measurement is
results/rq1_transformer_2026_reproduction/authors_code/logs/time_one_training_epoch.log.
"""
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

TRAINING_ROWS_PER_EPOCH = 77972   # 90 % of the 86,636 rows of the published resampled table


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    from tf_keras import layers
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=(128, 64),
                dropout=0.25, mlp_dropout=0.4):
    import tf_keras as keras
    from tf_keras import layers
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    return keras.Model(inputs, layers.Dense(1, activation="sigmoid")(x))


def main():
    import tensorflow as tf
    import tf_keras as keras
    print("TF threads:", tf.config.threading.get_intra_op_parallelism_threads(), "CPUs:", os.cpu_count(), flush=True)
    # random inputs with about the published share of failures; only the time per step matters here
    X = np.random.rand(512 * 40, 142, 1).astype(np.float32)
    y = (np.random.rand(512 * 40) < 0.34).astype(np.float32)
    for batch_size in [72, 256, 512]:
        model = build_model((142, 1))
        model.compile(optimizer=keras.optimizers.Adam(5e-4), loss="binary_crossentropy")
        steps = min(40, len(X) // batch_size)
        # a short first fit for warm-up and graph tracing, which would otherwise distort the timing
        model.fit(X[:batch_size * 2], y[:batch_size * 2], batch_size=batch_size, epochs=1, verbose=0)
        started = time.time()
        model.fit(X[:batch_size * steps], y[:batch_size * steps], batch_size=batch_size, epochs=1, verbose=0)
        seconds_per_step = (time.time() - started) / steps
        steps_per_epoch = int(np.ceil(TRAINING_ROWS_PER_EPOCH / batch_size))
        print(f"batch {batch_size}: {seconds_per_step * 1000:.1f} ms/step on this {os.cpu_count()}-vCPU CPU -> "
              f"{steps_per_epoch} steps/epoch -> est. {seconds_per_step * steps_per_epoch:.0f} s/epoch; "
              f"params {model.count_params()}", flush=True)


if __name__ == "__main__":
    main()
