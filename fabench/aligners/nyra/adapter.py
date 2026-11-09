# Copyright 2026 Olewave, LLC
# SPDX-License-Identifier: Apache-2.0

"""nyra forced aligner — word and phone forced alignment, via subprocess.

https://huggingface.co/nyralabs/nyra_forced_aligner_en
https://github.com/nyrahealth/nyra-forced-aligner

WavLM-large features projected by an LDA onto 40 dimensions, a triphone GMM-HMM
over 39 English phones plus 15 vocal-event units (fillers, laughter, breaths,
...), and a beam Viterbi over the transcript's graph. It is given the reference
transcript, so it belongs in track 1 beside MFA. It emits both tiers: words,
and the ARPAbet phones inside them.

Each word carries the package's confidence (0-1, how well the audio under the
word fits it), passed through as the span's 4th element so the calibration
metrics see it. Phones carry none.

Runs `evals/aligners/nyra/worker.py` in the tool's own interpreter, batched so
the model loads once per cell.
"""
from __future__ import annotations

from fabench.aligners.subprocess_aligner import SubprocessAligner


class Nyra(SubprocessAligner):
    source = "arpabet"
    emits_confidence = True
    granularity = ("word", "phone")
    default_model = "nyralabs/nyra_forced_aligner_en"

    def supports(self, mode: str, granularity: str) -> bool:
        # Text-driven only: the model builds its own phone graph from the words
        # and cannot consume a gold phone string (Mode B).
        return mode == "A" and super().supports(mode, granularity)

    def _extra_argv(self) -> list[str]:
        # WavLM precision. fp32 is the package's reference path; fp16 is its
        # CUDA default and moves <= 30 ms on ~0.5% of word boundaries.
        return [str(self.params.get("precision", "fp32"))]
