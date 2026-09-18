"""Qwen pause anchors must allow audible word tails before MFA cropping."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from adjust_ctc_boundaries import adjust_boundaries
from qwen3_timestamp_normalization import SCHEMA


def bundle(label='，', next_start=1.2):
    tokens = [dict(word='jia1', start_s=.2, end_s=.4, provider='qwen3_hf',
                   timestamp_normalization={'schema': SCHEMA}, raw_end_s=.4),
              dict(word='yi1', start_s=next_start, end_s=next_start+.2)]
    punct = [dict(word=label, start_s=.4, end_s=next_start,
                  raw_start_s=.4, raw_end_s=next_start, provider='qwen3_hf',
                  normalization_schema=SCHEMA,
                  left_lexical_ordinal=0, right_lexical_ordinal=1)]
    return tokens, punct


def audio(tail_end=.725, next_start=1.2, later_burst=False):
    sr = 16000
    times = np.arange(sr * 2) / sr
    result = np.full(len(times), .00005)
    mask = ((times >= .2) & (times < tail_end)) | ((times >= next_start) & (times < next_start+.2))
    if later_burst:
        mask |= (times >= .6) & (times < .8)
    result[mask] += .04 * np.sin(2 * np.pi * 200 * times[mask])
    return result, sr


@pytest.mark.parametrize('label', ['，', '。', '…'])
def test_audible_tail_moves_word_end_and_pause_start_together(label):
    tokens, punct = bundle(label)
    original = copy.deepcopy(punct)
    adjusted, marks, stats = adjust_boundaries(tokens, punct, *audio())
    assert adjusted[0]['end_s'] == pytest.approx(.725, abs=.005)
    assert marks[0]['start_s'] == adjusted[0]['end_s']
    assert marks[0]['word'] == label and marks[0]['end_s'] == 1.2
    assert adjusted[0]['raw_end_s'] == .4
    assert marks[0]['raw_start_s'] == original[0]['raw_start_s']
    assert stats['end_extend'] >= 1


@pytest.mark.parametrize('later_burst', [False, True])
def test_true_silence_does_not_extend_to_later_energy(later_burst):
    tokens, punct = bundle()
    adjusted, marks, _ = adjust_boundaries(tokens, punct, *audio(.4, later_burst=later_burst))
    assert adjusted[0]['end_s'] == .4
    assert marks[0]['start_s'] == .4


@pytest.mark.parametrize('invalid', ['provider', 'ordinal', 'schema'])
def test_other_or_unbound_punctuation_keeps_hard_boundary(invalid):
    tokens, punct = bundle()
    if invalid == 'provider':
        punct[0]['provider'] = 'nvasr'
    elif invalid == 'ordinal':
        punct[0]['left_lexical_ordinal'] = 9
    else:
        punct[0]['normalization_schema'] = 'unknown'
    adjusted, marks, _ = adjust_boundaries(tokens, punct, *audio(.6))
    assert adjusted[0]['end_s'] == .4
    assert marks[0]['start_s'] == .4


def test_extension_preserves_pause_and_does_not_cross_next_word():
    tokens, punct = bundle(next_start=.7)
    adjusted, marks, _ = adjust_boundaries(tokens, punct, *audio(.65, next_start=.7))
    assert .4 < adjusted[0]['end_s'] <= .64 + 1e-9
    assert marks[0]['end_s'] - marks[0]['start_s'] >= .06 - 1e-9
    assert adjusted[1]['start_s'] >= .7


@pytest.mark.parametrize('module', ['adjust_ctc_boundaries', 'run_pipeline'])
def test_energy_cache_requires_actual_execution_even_for_chinese(tmp_path, module):
    import importlib
    import json
    target = importlib.import_module(module)
    path = tmp_path / 'a_tokens.jsonl'
    row = {'word': 'jia1', 'start_s': .2, 'end_s': .4}
    path.write_text(json.dumps(row)+'\n')
    assert not target._processed_geometry_cache_complete(tmp_path, {'a'}, require_energy=True)
    row['energy_adjustment_schema'] = 'ctc-energy-adjustment-v1'
    path.write_text(json.dumps(row)+'\n')
    assert target._processed_geometry_cache_complete(tmp_path, {'a'}, require_energy=True)
    path.write_text('')
    assert not target._processed_geometry_cache_complete(tmp_path, {'a'}, require_energy=True)
