# Copyright 2026 Olewave, LLC
# SPDX-License-Identifier: Apache-2.0

"""nyra adapter: registry wiring and the worker protocol, without the model."""
from __future__ import annotations

import json
import sys

import pytest

from fabench.aligners import get_adapter
from fabench.aligners.base import AlignerError, BatchItem
from fabench.config import AlignerSpec


def _adapter(params: dict):
    return get_adapter(
        AlignerSpec(
            name="nyra",
            adapter="nyra",
            enabled=True,
            modes=["A"],
            granularity=["word", "phone"],
            emits_confidence=True,
            params=params,
        )
    )


def test_registry_entry_is_a_batch_word_and_phone_aligner():
    adapter = _adapter({"venv": "unused"})
    assert adapter.batch
    assert adapter.source == "arpabet"
    assert adapter.emits_confidence
    assert adapter.supports("A", "word")
    assert adapter.supports("A", "phone")
    assert not adapter.supports("B", "phone")


def test_load_requires_the_private_venv(tmp_path):
    with pytest.raises(AlignerError):
        _adapter({}).load()


def test_words_carry_confidence_and_phones_are_parsed(tmp_path):
    """A stand-in worker speaking nyra's protocol, run through the real adapter."""
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    record = {
        "item_id": "u1",
        "words": [["So,", 0.1, 0.3, 0.91], ["th-", 0.3, 0.35, 0.34]],
        "phones": [["S", 0.1, 0.2], ["OW", 0.2, 0.3], ["TH", 0.3, 0.35]],
    }
    (tmp_path / "worker.py").write_text(
        f"import sys\nprint({json.dumps(json.dumps(record))})\n"
    )
    adapter = _adapter({"venv": str(tmp_path / "venv")})

    out = adapter.align_corpus([BatchItem("u1", "a.wav", "so th-", "spk", None, "A")])["u1"]

    assert [(w.label, w.conf) for w in out.words] == [("so", 0.91), ("th-", 0.34)]
    assert [p.label for p in out.phones] == ["S", "OW", "TH"]
    assert all(p.conf is None for p in out.phones)
