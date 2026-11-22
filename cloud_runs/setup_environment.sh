#!/usr/bin/env bash
# Prepare a Colab runtime for the queue: the data, the Python packages, the external repositories and the exact
# inputs of the code released by Beikmohammadi et al. The second cell of a session notebook runs it from the repository
# root:
#
#   bash cloud_runs/setup_environment.sh
#
# It needs APS_CLOUD_STORE (the store folder in Google Drive) and APS_SESSION_NAME in the environment; the first cell
# of the notebook sets both. Everything it prints is also appended to <store>/logs/environment_<session>.log, so a
# failed step can be read in Drive. No step stops the script; the import check at the end says what is usable.
# The NVIDIA MPS daemon of the GPU sessions is started by the first cell (cloud_runs/colab/colab_bootstrap.py).
REPOSITORY_ROOT=$(cd "$(dirname "$0")/.." && pwd)
STORE=${APS_CLOUD_STORE:?set APS_CLOUD_STORE to the store folder}
SESSION=${APS_SESSION_NAME:-session}
mkdir -p "$STORE/logs"
exec > >(tee -a "$STORE/logs/environment_${SESSION}.log") 2>&1
echo "===== $(date -u +%FT%TZ) setup_environment.sh session=${SESSION} python=$(python -V 2>&1) ====="
cd "$REPOSITORY_ROOT" || { echo "no repository at $REPOSITORY_ROOT"; exit 2; }

# ---- the two UCI data files: from the store if copies are there, otherwise from the UCI repository
mkdir -p data
if [ -f "$STORE/inputs/data/aps_failure_training_set.csv" ]; then
  cp "$STORE"/inputs/data/aps_failure_*.csv data/ && echo "data files copied from the store"
else
  echo "data files not in the store; downloading them from the UCI repository"
  mkdir -p cache/uci_download
  wget -q -O cache/uci_download/aps.zip "https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip" \
    && unzip -o -q cache/uci_download/aps.zip -d cache/uci_download
  find cache/uci_download -name "aps_failure_training_set.csv" -exec cp {} data/ \;
  find cache/uci_download -name "aps_failure_test_set.csv" -exec cp {} data/ \;
fi
sha256sum data/aps_failure_*.csv     # compare with data/README.md
python -c "from experiments.common.data_loading import load_test_set, load_training_set; \
X, y = load_training_set(); X_test, y_test = load_test_set(); \
print('data ok', X.shape, int(y.sum()), X_test.shape, int(y_test.sum()))" || echo "DATA CHECK FAILED"

# ---- Python packages on top of the Colab image, installed in this order and in separate pip calls, as in the
# sessions that produced the results. pip reports a conflict between the huggingface-hub pin of tabpfn 2.1.3 and the
# huggingface-hub 1.x of the November 2026 image; TabPFN worked regardless.
echo "--- pip: requirements-colab.txt (tree learners, resampling, PyTorch, TabICL, TabPFN, TabEBM, Forest-Flow, TensorFlow)"
python -m pip install -q -r cloud_runs/requirements-colab.txt 2>&1 | grep -v "^$" | tail -20
echo "--- pip: CTGAN/TVAE, Gaussian copula, GReaT with LoRA, Unmasking Trees"
python -m pip install -q ctgan copulas be-great peft utrees 2>&1 | grep -v "^$" | tail -20
echo "--- pip: packages of the training code of Beikmohammadi et al. (focal loss; scikit-plot for their test())"
python -m pip install -q focal-loss scikit-plot 2>&1 | grep -v "^$" | tail -10
python - <<'EOF'
import importlib

failed = []
for name in ["numpy", "pandas", "sklearn", "lightgbm", "xgboost", "catboost", "imblearn", "optuna", "torch", "tabpfn",
             "tabicl", "tabebm", "ForestDiffusion", "ctgan", "copulas", "be_great", "peft", "utrees", "transformers"]:
    try:
        module = importlib.import_module(name)
        print(f"ok    {name:16s} {getattr(module, '__version__', '')}")
    except Exception as error:  # noqa: BLE001
        failed.append(name)
        print(f"FAIL  {name:16s} {type(error).__name__}: {str(error)[:140]}")
try:
    import torch
    print("torch cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception as error:  # noqa: BLE001
    print("torch check failed:", error)
print("IMPORT CHECK:", "all ok" if not failed else "FAILED " + ",".join(failed))
EOF
# the authors wrote for Keras 2: Colab ships TensorFlow with tf_keras, so their code runs in the main environment
TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=3 python -c "import tensorflow as tf, tf_keras, sklearn, imblearn, focal_loss; \
print('TF', tf.__version__, 'tf_keras', tf_keras.__version__, 'focal_loss ok', 'GPUs', tf.config.list_physical_devices('GPU'))" \
  2>&1 | grep -v "^I0000\|^WARNING\|oneDNN\|AVX" || echo "TF CHECK FAILED"

# ---- the external repositories at the commits used for the thesis (the authors' code, CTAB-GAN+)
bash third_party/fetch_third_party_code.sh
# ---- the six exact input tables of the authors' training code, if the store has them already
if [ -f "$STORE/inputs/authors_exact_inputs.zip" ]; then
  unzip -o -q "$STORE/inputs/authors_exact_inputs.zip" -d third_party/cost-sensitive-transformer-model/ \
    && echo "exact inputs unpacked: $(ls third_party/cost-sensitive-transformer-model/aps_failure_*.csv | wc -l) tables"
else
  echo "exact inputs not in the store yet (the job prepare_authors_exact_inputs makes them)"
fi
echo "===== setup_environment.sh done $(date -u +%FT%TZ) ====="
