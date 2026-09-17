from copy import deepcopy

import pytest

from scripts.ctc_prealign import attach_nvasr_candidate_provenance


@pytest.mark.parametrize("label", ["BREATHING", "COUGH", "LAUGHTER"])
def test_leading_nvv_without_candidate_is_rejected_without_mutating_rows(label):
    """A warm-up-prefix NVV cannot be silently removed without evidence."""
    words = [
        {"word": label, "start": 0.00, "end": 0.12},
        {"word": "ni3", "start": 0.12, "end": 0.30},
    ]
    before = deepcopy(words)

    errors = attach_nvasr_candidate_provenance(
        words, [], {"candidates": []})

    assert errors == [
        f"output NVV row 0: missing candidate mapping for {label!r} "
        "at neighbors (None, 0)"
    ]
    assert words == before
