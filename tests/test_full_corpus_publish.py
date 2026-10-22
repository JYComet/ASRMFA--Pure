from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import numpy as np
import soundfile as sf

from scripts.full_corpus_publish import validate_chunk_result, publish_chunk, build_final_report


def _evidence(root: Path, mode="reference"):
    qwen = root / "manifest.json"
    qwen.write_text('[{"stem":"u1","provider":"qwen3_hf","timestamp_normalization":"qwen3-timestamp-normalization-v2","lexical_timing_source":"qwen3_forced_aligner_hf","_words":[{"text":"你","start":0,"end":1}],"duration_s":1,"text_normalized":"你好"}]')
    identity = root / "identity.json"
    identity.write_text('{"schema":"qwen3-hf-identity-v1","provider":"qwen3_hf","runtime":{"python":"test"},"models":{"asr":"test"},"settings":{"batch_size":1},"inputs":{"stems":["u1"]},"identity_digest":"digest"}')
    raw = root / "raw.json"; raw.write_text('{}')
    accounting = root / "accounting.json"; accounting.write_text('{"schema":"pipeline-run-receipt-v2","run_health":"healthy","silent_loss":0,"source":{"count":1},"eligible":{"count":1,"stems":["u1"]},"output":{"count":1,"stems":["u1"]},"filtered":{"count":0,"stems":[]},"extra":{"reference_mode":"%s","identity_digest":"id","forced_aligner_model_tree_digest":"model"}}' % mode)
    strict = root / "strict.json"; strict.write_text('{}')
    if mode == "fallback":
        detail = '"fallback_transcript":"你好","fallback_punctuation_projection":{"verified":true}'
        contract = '"publication_contract":{"status":"verified"}'
    else:
        detail = '"timestamp_normalized_transcript":"你好"'
        contract = '"publication_contract":{"status":"verified","details":{"reference_punctuation_projection":{"verified":true}}}'
    report = root / "postprocess_report.jsonl"; report.write_text('{"stem":"u1","reference_mode":"%s","hard_integrity_reasons":[],%s,%s}\n' % (mode, detail, contract))
    paths = [qwen, identity, raw, accounting, report]
    return {"qwen_manifest": qwen, "qwen_identity": identity,
            "raw_timestamps": [raw], "accounting": accounting,
            "strict_manifest": strict, "postprocess_report": report,
            "punctuation_evidence": report,
            "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [qwen, identity, raw, accounting, strict, report]}}


def _grid(path: Path):
    from scripts.postprocess_textgrids import Interval, Tier, TextGrid, write_textgrid
    path.parent.mkdir(parents=True, exist_ok=True)
    tiers = [Tier(name, 0, 1, [Interval(0, 1, "x")])
             for name in ("raw_text", "pinyin", "hanzi", "words", "pinyin_phones")]
    write_textgrid(TextGrid(0, 1, tiers), path)


def test_publication_reclassifies_flat_results_and_conserves_denominator(tmp_path):
    accepted = tmp_path / "accepted" / "u1.TextGrid"; filtered = tmp_path / "filtered"
    _grid(accepted)
    wav = tmp_path / "u1.wav"; sf.write(wav, np.ones(16000) * .1, 16000, subtype="PCM_16")
    item = {"run_stem": "u1", "speaker": "今汐", "source_id": "wuwa",
            "game": "鸣潮", "text_mode": "reference", "pipeline_wav": wav}
    publication = validate_chunk_result({"chunk_id": "c1", "items": [item],
                                         "output_root": tmp_path / "accepted",
                                         "filtered_root": filtered,
                                         "evidence": _evidence(tmp_path)}, {"items": [item]})
    assert publication.accepted_stems == ("u1",)
    assert publication.accepted[0].speaker == "MC今汐"
    receipt = publish_chunk(publication, tmp_path / "0915ALL")
    assert Path(receipt["replacements"][0]["target"]).parent.name == "MC今汐"
    assert set(publication.accepted_stems) | set(publication.filtered_stems) | set(publication.failed_stems) == {"u1"}


def test_publish_replaces_only_exact_target_and_keeps_rollback(tmp_path):
    source = tmp_path / "source.TextGrid"; _grid(source)
    row = {"path": source, "speaker": "今汐"}
    target = tmp_path / "0915ALL" / "今汐" / source.name
    target.parent.mkdir(parents=True); target.write_text("old")
    receipt = publish_chunk({"accepted": [row], "filtered": []}, tmp_path / "0915ALL")
    assert receipt["replaced_count"] == 1
    assert Path(receipt["rollback_root"]).is_dir()


def test_missing_real_evidence_is_rejected(tmp_path):
    accepted = tmp_path / "accepted" / "u1.TextGrid"; _grid(accepted)
    wav = tmp_path / "u1.wav"; sf.write(wav, np.ones(16000) * .1, 16000, subtype="PCM_16")
    item = {"run_stem": "u1", "speaker": "今汐", "pipeline_wav": wav}
    try:
        validate_chunk_result({"items": [item], "output_root": accepted.parent,
                               "filtered_root": tmp_path / "filtered"}, {"items": [item]})
    except ValueError as exc:
        assert "evidence" in str(exc)
    else:
        raise AssertionError("caller booleans were accepted as evidence")


def test_v5_float_pipeline_axis_is_allowed(tmp_path):
    accepted = tmp_path / "accepted" / "u1.TextGrid"; _grid(accepted)
    wav = tmp_path / "u1.wav"; sf.write(wav, np.ones(16000, dtype="float32") * .1, 16000, subtype="FLOAT")
    item = {"run_stem": "u1", "speaker": "LAria", "source_id": "v5_0707",
            "text_mode": "fallback", "pipeline_wav": wav, "needs_gamesl_padding": False}
    publication = validate_chunk_result({"items": [item], "output_root": accepted.parent,
                                         "filtered_root": tmp_path / "filtered",
                                         "evidence": _evidence(tmp_path, "fallback")}, {"items": [item]})
    assert publication.accepted_stems == ("u1",)


def test_filtered_grid_uses_rejected_contract_as_filter_evidence(tmp_path):
    filtered = tmp_path / "filtered" / "u1.TextGrid"
    from scripts.postprocess_textgrids import Interval, Tier, TextGrid, write_textgrid
    filtered.parent.mkdir(parents=True, exist_ok=True)
    write_textgrid(TextGrid(0, 1, [Tier("words", 0, 1, [Interval(0, 1, "x")])]), filtered)
    wav = tmp_path / "u1.wav"
    sf.write(wav, np.ones(16000) * .1, 16000, subtype="PCM_16")
    item = {"run_stem": "u1", "speaker": "今汐", "source_id": "wuwa",
            "game": "鸣潮", "text_mode": "reference", "pipeline_wav": wav}
    evidence = _evidence(tmp_path)
    accounting = Path(evidence["accounting"])
    payload = json.loads(accounting.read_text())
    payload["output"] = {"count": 0, "stems": []}
    payload["filtered"] = {"count": 1, "stems": ["u1"]}
    accounting.write_text(json.dumps(payload))
    report = Path(evidence["postprocess_report"])
    report.write_text(json.dumps({
        "stem": "u1", "reference_mode": "reference",
        "status": "filtered_suspicious_alignment",
        "filter_reasons": ["suspicious_alignment"],
        "hard_integrity_reasons": ["reference_semantic_sequence_mismatch"],
        "publication_contract": {"status": "rejected", "reasons": ["owner_mismatch"]},
    }) + "\n")
    for path in (accounting, report):
        evidence["sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    publication = validate_chunk_result({
        "items": [item], "output_root": tmp_path / "accepted",
        "filtered_root": filtered.parent, "evidence": evidence,
    }, {"items": [item]})
    assert publication.filtered_stems == ("u1",)


def test_producer_filtered_item_can_be_explicitly_accounted_without_grid(tmp_path):
    accepted = tmp_path / "accepted" / "u1.TextGrid"
    _grid(accepted)
    items = []
    for stem in ("u1", "u2"):
        wav = tmp_path / f"{stem}.wav"
        sf.write(wav, np.ones(16000) * .1, 16000, subtype="PCM_16")
        items.append({"run_stem": stem, "speaker": "今汐", "source_id": "wuwa",
                      "game": "鸣潮", "text_mode": "reference", "pipeline_wav": wav})
    evidence = _evidence(tmp_path)
    evidence.update({"output_stems": ["u1"], "filtered_stems": ["u2"]})
    accounting = Path(evidence["accounting"])
    payload = json.loads(accounting.read_text())
    payload["source"] = {"count": 2, "stems": ["u1", "u2"]}
    payload["eligible"] = {"count": 2, "stems": ["u1", "u2"]}
    payload["output"] = {"count": 1, "stems": ["u1"]}
    payload["filtered"] = {"count": 1, "stems": ["u2"]}
    accounting.write_text(json.dumps(payload))
    evidence["sha256"][str(accounting)] = hashlib.sha256(accounting.read_bytes()).hexdigest()

    publication = validate_chunk_result({
        "items": items, "output_root": accepted.parent,
        "filtered_root": tmp_path / "filtered", "evidence": evidence,
        "failed": {"u2": "qwen_producer_filtered"},
    }, {"items": items})
    assert publication.accepted_stems == ("u1",)
    assert publication.failed_stems == ("u2",)


def test_textgrid_parser_imports_when_orchestrator_runs_as_direct_script(tmp_path):
    grid = tmp_path / "u1.TextGrid"
    _grid(grid)
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    env = dict(os.environ, PYTHONPATH=str(scripts))
    code = (
        "from pathlib import Path; "
        "from full_corpus_publish import _grid_tiers; "
        f"grid, tiers = _grid_tiers(Path({str(grid)!r})); "
        "assert len(tiers) == 5"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_chunk_publication_rolls_back_entire_group_and_can_retry(tmp_path, monkeypatch):
    first, second = tmp_path / "one.TextGrid", tmp_path / "two.TextGrid"
    first.write_text("new-one")
    second.write_text("new-two")
    out = tmp_path / "public"
    old_target = out / "speaker" / first.name
    old_target.parent.mkdir(parents=True)
    old_target.write_text("old-one")
    publication = {"accepted": [
        {"stem": "one", "path": first, "speaker": "speaker"},
        {"stem": "two", "path": second, "speaker": "speaker"},
    ], "filtered": []}
    real_replace = os.replace
    failed = False

    def fail_second_target(src, dst):
        nonlocal failed
        if Path(dst) == out / "speaker" / second.name and not failed:
            failed = True
            raise OSError("injected second target replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr("scripts.full_corpus_publish.os.replace", fail_second_target)
    try:
        publish_chunk(publication, out)
    except OSError as exc:
        assert "injected" in str(exc)
    else:
        raise AssertionError("injected transaction failure was ignored")
    assert old_target.read_text() == "old-one"
    assert not (out / "speaker" / second.name).exists()

    monkeypatch.setattr("scripts.full_corpus_publish.os.replace", real_replace)
    receipt = publish_chunk(publication, out)
    assert receipt["published_count"] == 2
    assert old_target.read_text() == "new-one"
    assert (out / "speaker" / second.name).read_text() == "new-two"


def test_chunk_publication_recovers_stranded_journal_before_retry(tmp_path):
    first, second = tmp_path / "one.TextGrid", tmp_path / "two.TextGrid"
    first.write_text("new-one")
    second.write_text("new-two")
    out = tmp_path / "public"
    old_target = out / "speaker" / first.name
    old_target.parent.mkdir(parents=True)
    old_target.write_text("old-one")
    publication = {"accepted": [
        {"stem": "one", "path": first, "speaker": "speaker"},
        {"stem": "two", "path": second, "speaker": "speaker"},
    ], "filtered": []}
    code = r'''
import os
from pathlib import Path
from scripts.full_corpus_publish import publish_chunk
import scripts.full_corpus_publish as pubmod
root = Path(os.environ["CASE_ROOT"])
first, second = root / "one.TextGrid", root / "two.TextGrid"
out = root / "public"
pub = {"accepted": [
    {"stem": "one", "path": first, "speaker": "speaker"},
    {"stem": "two", "path": second, "speaker": "speaker"}], "filtered": []}
real_replace = os.replace
def crash_after_first(src, dst):
    result = real_replace(src, dst)
    if Path(dst) == out / "speaker" / first.name:
        os._exit(23)
    return result
pubmod.os.replace = crash_after_first
publish_chunk(pub, out)
'''
    env = dict(os.environ, CASE_ROOT=str(tmp_path))
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], env=env)
    assert result.returncode == 23
    assert old_target.read_text() == "new-one"
    receipt = publish_chunk(publication, out)
    assert receipt["published_count"] == 2
    assert old_target.read_text() == "new-one"


def test_chunk_copy_failure_during_prepare_leaves_all_targets_untouched(tmp_path, monkeypatch):
    first, second = tmp_path / "one.TextGrid", tmp_path / "two.TextGrid"
    first.write_text("new-one")
    second.write_text("new-two")
    out = tmp_path / "public"
    target_one = out / "speaker" / first.name
    target_two = out / "speaker" / second.name
    target_one.parent.mkdir(parents=True)
    target_one.write_text("old-one")
    target_two.write_text("old-two")
    publication = {"accepted": [
        {"stem": "one", "path": first, "speaker": "speaker"},
        {"stem": "two", "path": second, "speaker": "speaker"},
    ], "filtered": []}
    real_copy = __import__("shutil").copyfile
    failed = False

    def fail_second_source(src, dst, *args, **kwargs):
        nonlocal failed
        if Path(src) == second and not failed:
            failed = True
            raise OSError("injected second source copy failure")
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr("scripts.full_corpus_transaction.shutil.copyfile", fail_second_source)
    try:
        publish_chunk(publication, out)
    except OSError as exc:
        assert "injected" in str(exc)
    else:
        raise AssertionError("injected prepare copy failure was ignored")
    assert target_one.read_text() == "old-one"
    assert target_two.read_text() == "old-two"
