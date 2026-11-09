#!/usr/bin/env python
# Copyright 2026 Olewave, LLC
# SPDX-License-Identifier: Apache-2.0

"""Batch forced-alignment worker for nyralabs/nyra_forced_aligner_en.

Runs in the tool's OWN interpreter (evals/aligners/nyra/venv/bin/python), as
every subprocess aligner does: nyra-forced-aligner brings its own torch,
transformers and numba.

Protocol: JSONL job records on argv[1], JSONL results to stdout.
    in : {"item_id": ..., "audio_path": ..., "transcript": ...}
    out: {"item_id": ..., "words":  [[text, start, end, confidence], ...],
                          "phones": [[phone, start, end], ...],
                          "skipped": [token, ...]}                      on success
         {"item_id": ..., "error": "..."}                              on failure

Words are the transcript's tokens as given. Phones are the model's ARPAbet
phones (39 English phones plus AX), unstressed, covering each word, and `SIL`
for the silence it places before, between and after words, in time order (the
reference phone tiers carry silence too). `confidence` (0-1) is the package's per-word
fit of the audio to the word (nyra_align.confidence); it is per word, so the
phones carry none. `skipped` (only when non-empty) lists tokens the aligner
could not place and survives into hyp.jsonl as diagnostics.

The model loads ONCE for the whole batch.
"""
import json
import sys

from nyra_align import Aligner


def main() -> int:
    with open(sys.argv[1]) as job_file:
        jobs = [json.loads(line) for line in job_file if line.strip()]
    model = sys.argv[2] if len(sys.argv) > 2 else "nyralabs/nyra_forced_aligner_en"
    precision = sys.argv[3] if len(sys.argv) > 3 else "fp32"
    aligner = Aligner(model, precision=precision)

    for job in jobs:
        try:
            result = aligner.align(job["audio_path"], job["transcript"])
            words = [
                [w.word, float(w.start), float(w.end), float(w.confidence)]
                for w in result.words
            ]
            phones = sorted(
                [[p.phone, float(p.start), float(p.end)] for w in result.words for p in w.phones]
                + [[s.phone, float(s.start), float(s.end)] for s in result.silences],
                key=lambda span: span[1],
            )
            record = {"item_id": job["item_id"], "words": words, "phones": phones}
            if result.skipped:
                record["skipped"] = list(result.skipped)
            print(json.dumps(record), flush=True)
        except Exception as exc:  # one bad item must not lose the batch
            print(json.dumps({"item_id": job["item_id"], "error": str(exc)[:300]}),
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
