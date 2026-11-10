#!/usr/bin/env bash
# Copyright 2026 Olewave, LLC
# SPDX-License-Identifier: Apache-2.0

# nyra forced aligner for FA-Bench -- PRIVATE venv.
#
#   https://huggingface.co/nyralabs/nyra_forced_aligner_en
#   https://pypi.org/project/nyra-forced-aligner/
#
# Own venv, like every subprocess aligner: the package brings its own torch,
# transformers and numba. espeak-ng is only needed for words outside the
# model's 41k-word lexicon (`apt install espeak-ng`).
#
# TORCH BUILD: cu128, equal to the shared .venv's torch.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL=${MODEL:-nyralabs/nyra_forced_aligner_en}

uv venv "$HERE/venv" --python 3.12
uv pip install --python "$HERE/venv/bin/python" \
    torch==2.8.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple
# VERSION PINNED to what produced the published numbers. 0.2.3 is the first
# release whose long-audio window stitching does not drift (0.2.2 lost 20 ms
# every 29 s on recordings over 30 s) and the first with per-word confidence;
# 0.2.5 stops leading silence sticking to the first word.
uv pip install --python "$HERE/venv/bin/python" nyra-forced-aligner==0.2.5

"$HERE/venv/bin/python" - <<PY
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
from nyra_align import Aligner
print("fetching $MODEL ...")
Aligner("$MODEL")
print("nyra ok")
PY
