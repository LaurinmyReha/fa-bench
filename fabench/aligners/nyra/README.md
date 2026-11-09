# nyra forced aligner

[nyra_forced_aligner_en](https://huggingface.co/nyralabs/nyra_forced_aligner_en)
([nyra-forced-aligner](https://github.com/nyrahealth/nyra-forced-aligner) on
PyPI) is a verbatim-aware forced aligner for conversational English: WavLM-large
features projected by an LDA, a triphone GMM-HMM over the 39 English phones
plus 15 vocal-event units (fillers, laughter, breaths, coughs, ...), and a beam
Viterbi over the transcript. It is given the words, so it runs in track 1
beside MFA and Olign.

- **Modes.** A (text-driven). **Granularity.** Word and phone: the phones are
  the model's ARPAbet units inside each word, normalized as `arpabet`
  (`AX` folds to `ah`). **Confidence.** Per word, 0 to 1: how well the audio
  under the word fits it. Phones carry none.

## Requirements

```bash
evals/aligners/nyra/download_and_install.sh    # own venv, nyra-forced-aligner==0.2.4
```

A private venv, like every subprocess aligner: the package brings its own torch,
transformers and numba. The adapter runs `worker.py` in that interpreter,
batched so the model loads once per cell. `espeak-ng` is only needed for words
outside the model's 41k-word lexicon.

## Config

```yaml
- name: nyra
  adapter: nyra
  enabled: true
  modes: [A]
  granularity: [word, phone]
  emits_confidence: true
  params:
    venv: venv
    model: nyralabs/nyra_forced_aligner_en
    precision: fp32      # reference path; fp16 is ~2x faster, <= 30 ms on ~0.5% of boundaries
    timeout_s: null      # results arrive when the batch finishes; a timeout loses the cell
```

## Caveats

- **Use 0.2.3 or later.** Up to 0.2.2 the package's 30 s WavLM windows were
  stitched one frame short each, so recordings longer than 30 s drifted by
  20 ms every 29 s. TIMIT and Buckeye test utterances are shorter than that and
  were never affected, but the pin keeps longer inputs honest.
- **Silences are not reported** as phones: the phone tier covers the words'
  spans only. Event units (`LAU`, `FUH`, ...) appear only for bracketed tags in
  a transcript, which the TIMIT and Buckeye references do not use.
- **Licence.** The package is MIT. The model, and the alignments it produces,
  are under the nyra health Non-Commercial Research License: free for research
  and other non-commercial use, commercial use needs a licence from nyra health.
