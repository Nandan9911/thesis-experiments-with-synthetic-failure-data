#!/usr/bin/env bash
# Clone the two external repositories that some experiments need, at the exact commits used for the thesis.
#
#   cost-sensitive-transformer-model  code, preprocessing script and trained models of Beikmohammadi et al. (2026);
#                                     needed only for RQ1b (experiments/rq1_transformer_2026_reproduction/authors_code)
#                                     and for the RQ1b cloud jobs. About 100 MB.
#   CTAB-GAN-Plus                     the reference implementation of CTAB-GAN+ by Zhao et al.; needed only for the
#                                     RQ2 arm lightgbm_with_ctab_gan_plus_rows and its diagnostics.
#
# Run from anywhere:  bash third_party/fetch_third_party_code.sh
# Running it again is harmless: an existing clone is only checked out at the pinned commit.
set -euo pipefail
cd "$(dirname "$0")"

fetch() {
    local url="$1" folder="$2" commit="$3"
    if [ -d "$folder/.git" ]; then
        echo "$folder: already cloned"
    else
        git clone --quiet "$url" "$folder"
    fi
    git -C "$folder" checkout --quiet "$commit"
    echo "$folder: at commit $(git -C "$folder" rev-parse --short HEAD)"
}

fetch https://github.com/AliBeikmohammadi/Cost-Sensitive-Transformer-Model cost-sensitive-transformer-model \
      4db444a3958f0a5e95536e29b56206c497bb6d3d
fetch https://github.com/Team-TUD/CTAB-GAN-Plus CTAB-GAN-Plus \
      6a6f90188cca3dac2c533fd5e8e7f20de074365b
