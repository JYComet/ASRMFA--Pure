#!/usr/bin/env python3
"""Synthetic strict-en-mfa-v1 checks; no real MFA process is invoked."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import align_english_mfa as align
import postprocess_textgrids as post
from postprocess_textgrids import Interval, TextGrid, Tier, write_textgrid


def _tg(path, tiers):
    write_textgrid(TextGrid(0, 2, tiers), path)


def _segments(word="oovword"):
    return {"x": [{"seg_idx": 0, "segment_ordinal": 0, "seg_start": 0,
                    "seg_end": 1, "words": [{"text": word, "start": 0,
                    "end": 1, "ordinal": 0}]}]}


def _expect(condition, label):
    if condition:
        print(f"OK {label}")
        return 0
    print(f"FAIL {label}")
    return 1


def _g2p_failure_cases(root):
    fails = 0
    base = root / "base.dict"; base.write_text("known K N OW N\n")
    g2p = root / "g2p.zip"; g2p.write_bytes(b"fixture")

    def invoke(kind):
        with tempfile.TemporaryDirectory(dir=root) as td:
            work = Path(td)
            if kind == "missing":
                return align.build_en_dict(_segments(), base, work / "no-model", Path("python"),
                                           work, work, 1, strict=True)
            if kind == "missing_argument":
                return align.build_en_dict(_segments(), base, Path(""), Path("python"),
                                           work, work, 1, strict=True)
            def fake_run(*args, **kwargs):
                out = work / "en_oov_dict.txt"
                if kind == "timeout":
                    raise subprocess.TimeoutExpired(args[0], 1)
                if kind == "launch":
                    raise OSError("launch fixture")
                if kind == "nonzero":
                    return type("R", (), {"returncode": 9, "stderr": "bad fixture"})()
                if kind == "empty":
                    out.write_text("")
                elif kind == "incomplete":
                    out.write_text("different D IH F ER AH N T\n")
                # missing intentionally leaves no output
                return type("R", (), {"returncode": 0, "stderr": ""})()
            with patch.object(align.subprocess, "run", side_effect=fake_run):
                return align.build_en_dict(_segments(), base, g2p, Path("python"), work,
                                           work, 1, strict=True)

    for kind in ("missing", "missing_argument", "timeout", "launch", "nonzero", "empty", "incomplete"):
        try:
            invoke(kind)
            fails += _expect(False, f"strict G2P {kind}")
        except align.StrictG2PError:
            fails += _expect(True, f"strict G2P {kind}")
    return fails


def _corpus_cases(root):
    fails = 0
    missing = _segments()
    align.build_en_corpus(missing, root / "no-audio", root / "corpus-missing", strict=True)
    fails += _expect(missing["x"][0].get("reject_reason") == "audio_missing",
                     "strict audio missing retains denominator")

    unreadable = _segments()
    audio = root / "audio"; audio.mkdir(); (audio / "x.wav").write_bytes(b"not wav")
    with patch("scipy.io.wavfile.read", side_effect=ValueError("bad wav")):
        align.build_en_corpus(unreadable, audio, root / "corpus-bad", strict=True)
    fails += _expect(unreadable["x"][0].get("reject_reason") == "audio_unreadable",
                     "strict audio unreadable retains denominator")

    worker = _segments()
    with patch.object(align, "_build_corpus_stem", side_effect=RuntimeError("worker fixture")):
        align.build_en_corpus(worker, audio, root / "corpus-worker", strict=True)
    fails += _expect(worker["x"][0].get("reject_reason") == "corpus_worker_error",
                     "strict corpus worker error retains denominator")
    short = _segments(); short["x"][0]["seg_end"] = .01
    with patch("scipy.io.wavfile.read", return_value=(16000, __import__("numpy").zeros(16000, dtype="int16"))):
        align.build_en_corpus(short, audio, root / "corpus-short", strict=True)
    fails += _expect(short["x"][0].get("reject_reason") == "segment_too_short",
                     "strict short segment reason")

    # The English-MFA attempt floor is 150 ms.  This is only an attempt
    # eligibility gate; strict provenance still decides publication after MFA.
    at_floor = _segments(); at_floor["x"][0]["seg_end"] = .150
    with patch("scipy.io.wavfile.read", return_value=(16000, __import__("numpy").zeros(16000, dtype="int16"))):
        align.build_en_corpus(at_floor, audio, root / "corpus-at-floor", strict=True)
    fails += _expect(not at_floor["x"][0].get("skipped")
                     and (root / "corpus-at-floor" / "x_seg0.wav").is_file(),
                     "150ms segment is attempted")

    below_floor = _segments(); below_floor["x"][0]["seg_end"] = .149
    with patch("scipy.io.wavfile.read", return_value=(16000, __import__("numpy").zeros(16000, dtype="int16"))):
        align.build_en_corpus(below_floor, audio, root / "corpus-below-floor", strict=True)
    fails += _expect(below_floor["x"][0].get("reject_reason") == "segment_too_short",
                     "below-150ms segment is rejected")
    return fails


def _strict_mfa_runner_case(root):
    """Exercise strict command/log handling without invoking MFA."""
    corpus = root / "runner-corpus"; output = root / "runner-out"; work = root / "runner-work"
    corpus.mkdir(); dictionary = root / "runner.dict"; dictionary.write_text("hello HH AH L OW\n")
    model = root / "runner-model.zip"; model.write_bytes(b"model")
    completed = type("R", (), {"returncode": 0})()
    with patch.object(align.subprocess, "run", return_value=completed) as run:
        outcome = align.run_en_mfa(corpus, dictionary, str(model), output, work, Path("python"), root,
                                   strict=True, timeout=9)
    log = Path(outcome["log_path"])
    kwargs = run.call_args.kwargs
    good = ("--clean" not in outcome["command"] and "capture_output" not in kwargs
            and kwargs.get("stderr") is align.subprocess.STDOUT and getattr(kwargs.get("stdout"), "name", None) == str(log)
            and log.is_file() and "command: " in log.read_text() and "[outcome]" in log.read_text())
    fails = _expect(good, "strict MFA streams output to retained command/footer log")
    timeout = subprocess.TimeoutExpired(["fixture"], 9, output=b"large bytes", stderr=b"more bytes")
    with patch.object(align.subprocess, "run", side_effect=timeout):
        timed_out = align.run_en_mfa(corpus, dictionary, str(model), output, root / "runner-timeout", Path("python"), root,
                                     strict=True, timeout=9)
    timeout_log = Path(timed_out["log_path"])
    fails += _expect(timed_out["return_code"] == "timeout" and timed_out["timed_out"]
                     and "return_code: timeout" in timeout_log.read_text(),
                     "strict MFA timeout remains structured with streamed-log footer")
    return fails


def _main_failure_cases(root):
    fails = 0
    ctc = root / "ctc"; ctc.mkdir()
    _tg(ctc / "x.TextGrid", [Tier("words", 0, 2, [Interval(0, 1, "hello")])])
    (ctc / "x.lab").write_text("hello\n")
    dictionary = root / "dict"; dictionary.write_text("hello HH AH L OW\n")
    model = root / "model.zip"; model.write_bytes(b"model")
    fake_python = root / "python"; fake_python.write_text("")

    outcomes = {
        "timeout": {"return_code": "timeout", "timed_out": True, "timeout_seconds": 7,
                    "command": ["fixture", "align"], "acoustic_model": str(model), "exception": "", "log_path": "fixture.log"},
        "nonzero": {"return_code": 3, "timed_out": False, "timeout_seconds": 7,
                    "command": ["fixture", "align"], "acoustic_model": str(model), "exception": "", "log_path": "fixture.log"},
        "launch": {"return_code": "exception", "timed_out": False, "timeout_seconds": 7,
                   "command": ["fixture", "align"], "acoustic_model": str(model), "exception": "launch fixture", "log_path": "fixture.log"},
    }
    for label, outcome in outcomes.items():
        out = root / f"out-{label}"; temp = root / f"temp-{label}"
        argv = ["align_english_mfa.py", "--ctc-dir", str(ctc), "--audio-dir", str(root / "audio-none"),
                "--output-dir", str(out), "--acoustic-model", str(model), "--dictionary", str(dictionary),
                "--temp-dir", str(temp), "--python", str(fake_python), "--strict-provenance", "--timeout", "7"]
        with patch.object(sys, "argv", argv), patch.object(align, "build_en_corpus", side_effect=lambda x, *a, **k: x), \
             patch.object(align, "run_en_mfa", return_value=outcome):
            rc = align.main()
        manifest = json.loads((out / "en_alignment_manifest.json").read_text())
        artifact_dirs = all((temp / name).is_dir() for name in ("en_corpus", "en_aligned", "en_mfa_work", "log"))
        good = (rc == 1 and manifest["status"] == "failed" and manifest["expected_segments"] == ["x:s0"]
                and manifest["counts"] == {"english_stems": 1, "english_segments": 1,
                                           "english_words": 1, "verified_words": 0, "rejected_words": 1}
                and manifest["mfa"]["command"] == outcome["command"]
                and manifest["mfa"]["return_code"] == outcome["return_code"]
                and manifest["mfa"]["log_path"] == "fixture.log"
                and manifest["mfa"]["acoustic_model_sha256"] and manifest["mfa"]["dictionary_sha256"]
                and artifact_dirs)
        fails += _expect(good, f"strict MFA {label} failed manifest and retained artifacts")

    # Dictionary construction failures must take the same frozen-manifest path
    # and must not reach the MFA launcher.
    g2p_out = root / "out-g2p"; g2p_temp = root / "temp-g2p"; empty_dict = root / "empty.dict"
    empty_dict.write_text("")
    argv = ["align_english_mfa.py", "--ctc-dir", str(ctc), "--audio-dir", str(root / "audio-none"),
            "--output-dir", str(g2p_out), "--acoustic-model", str(model), "--dictionary", str(empty_dict),
            "--g2p-model", str(root / "missing-g2p"), "--temp-dir", str(g2p_temp), "--python", str(fake_python),
            "--strict-provenance", "--timeout", "7"]
    with patch.object(sys, "argv", argv), patch.object(align, "build_en_corpus", side_effect=lambda x, *a, **k: x), \
         patch.object(align, "run_en_mfa", side_effect=AssertionError("MFA must not start after G2P failure")):
        rc = align.main()
    manifest = json.loads((g2p_out / "en_alignment_manifest.json").read_text())
    fails += _expect(rc == 1 and manifest["status"] == "failed" and manifest["expected_segments"] == ["x:s0"]
                     and manifest["counts"]["english_words"] == 1 and manifest["mfa"]["return_code"] == "not_run",
                     "strict G2P main failure manifest and no MFA launch")

    success_out = root / "out-success"; success_temp = root / "temp-success"
    argv = ["align_english_mfa.py", "--ctc-dir", str(ctc), "--audio-dir", str(root / "audio-none"),
            "--output-dir", str(success_out), "--acoustic-model", str(model), "--dictionary", str(dictionary),
            "--temp-dir", str(success_temp), "--python", str(fake_python), "--strict-provenance", "--timeout", "7"]
    success = {"return_code": 0, "timed_out": False, "timeout_seconds": 7,
               "command": ["fixture", "align"], "acoustic_model": str(model), "exception": ""}
    with patch.object(sys, "argv", argv), patch.object(align, "build_en_corpus", side_effect=lambda x, *a, **k: x), \
         patch.object(align, "run_en_mfa", return_value=success):
        rc = align.main()
    success_manifest = json.loads((success_out / "en_alignment_manifest.json").read_text())
    fails += _expect(rc == 0 and success_manifest["schema"] == align.STRICT_SCHEMA
                     and success_manifest["status"] == "success" and len(success_manifest["rejected_segments"]) == 1
                     and all((success_temp / name).is_dir() for name in
                             ("en_corpus", "en_aligned", "en_mfa_work", "log")),
                     "strict main accepts complete manifest with local rejection")

    # If every English segment is locally rejected before MFA, strict mode may
    # publish a successful rejected ledger only after hashing the exact model
    # and configured dictionary.  It must not silently turn a hash failure
    # into a success/rc=0 merely because no MFA subprocess is needed.
    def all_local_rejections(segments, *unused_args, **unused_kwargs):
        for stem_segments in segments.values():
            for segment in stem_segments:
                segment["skipped"] = True
                segment["reject_reason"] = "audio_missing"
        return segments

    zero_hash_out = root / "out-zero-hash"; zero_hash_temp = root / "temp-zero-hash"
    argv = ["align_english_mfa.py", "--ctc-dir", str(ctc), "--audio-dir", str(root / "audio-none"),
            "--output-dir", str(zero_hash_out), "--acoustic-model", str(model), "--dictionary", str(dictionary),
            "--temp-dir", str(zero_hash_temp), "--python", str(fake_python), "--strict-provenance"]
    with patch.object(sys, "argv", argv), patch.object(align, "build_en_corpus", side_effect=all_local_rejections), \
         patch.object(align, "_resolve_acoustic_model", return_value=model), \
         patch.object(align, "_provenance_sha256", side_effect=OSError("zero-segment hash fixture")), \
         patch.object(align, "run_en_mfa", side_effect=AssertionError("MFA must not start for local rejections")):
        rc = align.main()
    zero_hash_manifest = json.loads((zero_hash_out / "en_alignment_manifest.json").read_text())
    fails += _expect(rc == 1 and zero_hash_manifest["status"] == "failed"
                     and zero_hash_manifest["reason"] == "mfa_input_hash_failed"
                     and zero_hash_manifest["mfa"]["return_code"] == "not_run",
                     "zero runnable strict segments fail on model/dictionary hash")

    zero_ok_out = root / "out-zero-ok"; zero_ok_temp = root / "temp-zero-ok"
    argv = ["align_english_mfa.py", "--ctc-dir", str(ctc), "--audio-dir", str(root / "audio-none"),
            "--output-dir", str(zero_ok_out), "--acoustic-model", str(model), "--dictionary", str(dictionary),
            "--temp-dir", str(zero_ok_temp), "--python", str(fake_python), "--strict-provenance"]
    with patch.object(sys, "argv", argv), patch.object(align, "build_en_corpus", side_effect=all_local_rejections), \
         patch.object(align, "run_en_mfa", side_effect=AssertionError("MFA must not start for local rejections")):
        rc = align.main()
    zero_ok_manifest = json.loads((zero_ok_out / "en_alignment_manifest.json").read_text())
    fails += _expect(rc == 0 and zero_ok_manifest["status"] == "success"
                     and len(zero_ok_manifest["rejected_segments"]) == 1
                     and zero_ok_manifest["mfa"]["acoustic_model_sha256"]
                     and zero_ok_manifest["mfa"]["dictionary_sha256"],
                     "zero runnable strict segments succeed only with input hashes")

    # A producer failure is a process failure even when MFA itself completed.
    # The main-level gate must also reject unreadable and unknown-schema output.
    for label, text in (
        ("failed", json.dumps({"schema": align.STRICT_SCHEMA, "status": "failed"})),
        ("corrupt", "not json"),
        ("unknown-schema", json.dumps({"schema": "unknown", "status": "success"})),
    ):
        producer_out = root / f"out-producer-{label}"; producer_temp = root / f"temp-producer-{label}"
        argv = ["align_english_mfa.py", "--ctc-dir", str(ctc), "--audio-dir", str(root / "audio-none"),
                "--output-dir", str(producer_out), "--acoustic-model", str(model), "--dictionary", str(dictionary),
                "--temp-dir", str(producer_temp), "--python", str(fake_python), "--strict-provenance", "--timeout", "7"]
        def producer(*args, **kwargs):
            manifest_path = Path(args[3]) / "en_alignment_manifest.json"
            manifest_path.write_text(text, encoding="utf-8")
            return manifest_path
        with patch.object(sys, "argv", argv), patch.object(align, "build_en_corpus", side_effect=lambda x, *a, **k: x), \
             patch.object(align, "run_en_mfa", return_value=success), \
             patch.object(align, "produce_strict_ledgers", side_effect=producer):
            rc = align.main()
        fails += _expect(rc == 1, f"strict main rejects {label} producer manifest")

    # Hashing must be a pre-launch gate: no MFA invocation is permitted if the
    # exact resolved acoustic input cannot be hashed.
    hash_out = root / "out-hash"; hash_temp = root / "temp-hash"
    argv = ["align_english_mfa.py", "--ctc-dir", str(ctc), "--audio-dir", str(root / "audio-none"),
            "--output-dir", str(hash_out), "--acoustic-model", str(model), "--dictionary", str(dictionary),
            "--temp-dir", str(hash_temp), "--python", str(fake_python), "--strict-provenance"]
    original_hash = align._provenance_sha256
    def fail_model_hash(path):
        if Path(path) == model:
            raise OSError("model hash fixture")
        return original_hash(Path(path))
    with patch.object(sys, "argv", argv), patch.object(align, "build_en_corpus", side_effect=lambda x, *a, **k: x), \
         patch.object(align, "_resolve_acoustic_model", return_value=model), \
         patch.object(align, "_provenance_sha256", side_effect=fail_model_hash), \
         patch.object(align, "run_en_mfa", side_effect=AssertionError("MFA must not start before hashes")):
        rc = align.main()
    manifest = json.loads((hash_out / "en_alignment_manifest.json").read_text())
    fails += _expect(rc == 1 and manifest["status"] == "failed" and manifest["reason"] == "mfa_input_hash_failed"
                     and manifest["expected_segments"] == ["x:s0"] and not manifest["mfa"]["acoustic_model_sha256"],
                     "strict model hash failure blocks MFA launch")

    # No labs is an input failure; a lab containing only Chinese remains a
    # successful no_english outcome and must not require an MFA interpreter.
    empty_ctc = root / "empty-ctc"; empty_ctc.mkdir(); empty_out = root / "out-no-ctc"
    argv = ["align_english_mfa.py", "--ctc-dir", str(empty_ctc), "--audio-dir", str(root / "audio-none"),
            "--output-dir", str(empty_out), "--acoustic-model", str(model), "--dictionary", str(dictionary),
            "--strict-provenance"]
    with patch.object(sys, "argv", argv):
        rc = align.main()
    manifest = json.loads((empty_out / "en_alignment_manifest.json").read_text())
    fails += _expect(rc == 1 and manifest["status"] == "failed" and manifest["reason"] == "no_ctc_stems"
                     and manifest["counts"] == {"english_stems": 0, "english_segments": 0,
                                                "english_words": 0, "verified_words": 0, "rejected_words": 0},
                     "strict no CTC stems writes failed zero manifest")

    chinese_ctc = root / "chinese-ctc"; chinese_ctc.mkdir(); chinese_out = root / "out-chinese"
    _tg(chinese_ctc / "cn.TextGrid", [Tier("words", 0, 1, [Interval(0, 1, "ni3")])])
    (chinese_ctc / "cn.lab").write_text("ni3\n")
    argv = ["align_english_mfa.py", "--ctc-dir", str(chinese_ctc), "--audio-dir", str(root / "audio-none"),
            "--output-dir", str(chinese_out), "--acoustic-model", str(model), "--dictionary", str(dictionary),
            "--strict-provenance"]
    with patch.object(sys, "argv", argv), \
         patch.object(align, "_provenance_sha256", side_effect=AssertionError("all-Chinese must not hash English MFA inputs")):
        rc = align.main()
    manifest = json.loads((chinese_out / "en_alignment_manifest.json").read_text())
    fails += _expect(rc == 0 and manifest["status"] == "no_english"
                     and not manifest["mfa"]["acoustic_model_sha256"],
                     "strict all-Chinese remains no_english without model dependency")
    return fails


def _ledger_fixture(root, label, words=("hello",)):
    """Create one isolated synthetic CTC segment and its MFA output root."""
    case = root / f"ledger-{label}"; ctc = case / "ctc"; aligned = case / "aligned"; output = case / "out"
    ctc.mkdir(parents=True); aligned.mkdir(); output.mkdir()
    step = 1 / len(words)
    ctc_words = [Interval(index * step, (index + 1) * step, word)
                 for index, word in enumerate(words)]
    _tg(ctc / "x.TextGrid", [Tier("words", 0, 1, ctc_words)])
    segment = {"x": [{"seg_idx": 0, "segment_ordinal": 0, "seg_name": "x_seg0",
                       "words": [{"text": word, "start": index * step, "end": (index + 1) * step,
                                  "ordinal": index}
                                 for index, word in enumerate(words)]}]}
    return ctc, aligned, output, segment


def _ledger_manifest(ctc, aligned, output, segment):
    return json.loads(align.produce_strict_ledgers(segment, ctc, aligned, output,
                                                   {"return_code": 0}).read_text())


def _ledger_cases(root):
    """Matrix 7--16: every fixture is local and never launches MFA."""
    fails = 0

    def reject(label, tiers=None, words=("hello",), nested=False, skipped=False):
        nonlocal fails
        ctc, aligned, output, segment = _ledger_fixture(root, label, words)
        if skipped:
            segment["x"][0]["skipped"] = True; segment["x"][0]["reject_reason"] = "segment_too_short"
        elif tiers is not None:
            target = aligned / "x_seg0" / "x_seg0.TextGrid" if nested else aligned / "x_seg0.TextGrid"
            target.parent.mkdir(parents=True, exist_ok=True)
            _tg(target, tiers)
        manifest = _ledger_manifest(ctc, aligned, output, segment)
        ledger = json.loads((output / "x_en_phones.json").read_text())
        row = ledger["segments"][0]
        fails += _expect(manifest["rejected_segments"] == [{"id": "x:s0", "reason": row["reason"]}]
                         and row["status"] == "rejected"
                         and [word["word_id"] for word in row["words"]] ==
                         [f"x:s0:w{index}" for index in range(len(words))], label)
        return ctc, aligned, output, segment, manifest, ledger

    # Source availability and tier identity are all fail-closed.
    reject("short source", skipped=True)
    reject("missing source TG")
    ctc, aligned, output, segment = _ledger_fixture(root, "unreadable")
    (aligned / "x_seg0.TextGrid").write_text("not a TextGrid", encoding="utf-8")
    manifest = _ledger_manifest(ctc, aligned, output, segment)
    fails += _expect(manifest["rejected_segments"][0]["reason"].startswith("No tiers found"), "unreadable source TG")
    reject("missing phones tier", [Tier("words", 0, 1, [Interval(0, 1, "hello")])])
    reject("duplicate words tier", [Tier("words", 0, 1, [Interval(0, 1, "hello")]),
                                      Tier("words", 0, 1, [Interval(0, 1, "hello")]),
                                      Tier("phones", 0, 1, [Interval(0, 1, "HH")])])

    valid = [Tier("words", 0, 1, [Interval(0, 1, "hello")]),
             Tier("phones", 0, 1, [Interval(0, .5, "HH"), Interval(.5, 1, "AH")])]
    ctc, aligned, output, segment = _ledger_fixture(root, "nested")
    nested = aligned / "x_seg0" / "x_seg0.TextGrid"; nested.parent.mkdir(); _tg(nested, valid)
    manifest = _ledger_manifest(ctc, aligned, output, segment)
    fails += _expect(manifest["status"] == "success" and manifest["produced_segments"] == ["x:s0"],
                     "nested source TG")
    _tg(aligned / "x_seg0.TextGrid", valid)
    manifest = _ledger_manifest(ctc, aligned, output, segment)
    fails += _expect(manifest["rejected_segments"][0]["reason"] == "source_tg_missing_or_ambiguous",
                     "flat nested ambiguity")

    reject("extra source word", [Tier("words", 0, 1, [Interval(0, .5, "hello"), Interval(.5, 1, "extra")]),
                                  Tier("phones", 0, 1, [Interval(0, .5, "HH"), Interval(.5, 1, "AH")])])
    reject("missing source word", [Tier("words", 0, 1, []), Tier("phones", 0, 1, [])])
    reject("reordered source words", [Tier("words", 0, 1, [Interval(0, .5, "world"), Interval(.5, 1, "hello")]),
                                       Tier("phones", 0, 1, [Interval(0, .25, "HH"), Interval(.25, .5, "AH"),
                                                               Interval(.5, .75, "W"), Interval(.75, 1, "ER")])],
           words=("hello", "world"))
    reject("source text mismatch", [Tier("words", 0, 1, [Interval(0, 1, "hullo")]), Tier("phones", 0, 1, [Interval(0, 1, "HH")])])

    reject("empty phones", [Tier("words", 0, 1, [Interval(0, 1, "hello")]), Tier("phones", 0, 1, [Interval(0, 1, "spn")])])
    reject("silence cannot mask empty", [Tier("words", 0, 1, [Interval(0, 1, "hello")]), Tier("phones", 0, 1, [Interval(0, 1, "sp")])])
    reject("unknown phone", [Tier("words", 0, 1, [Interval(0, 1, "hello")]), Tier("phones", 0, 1, [Interval(0, 1, "NOTAPHONE")])])
    reject("negative phone", [Tier("words", 0, 1, [Interval(0, 1, "hello")]), Tier("phones", 0, 1, [Interval(.4, .2, "HH")])])
    reject("overlap phones", [Tier("words", 0, 1, [Interval(0, 1, "hello")]), Tier("phones", 0, 1, [Interval(0, .7, "HH"), Interval(.6, 1, "AH")])])
    reject("out of word phone", [Tier("words", 0, 1, [Interval(.1, .9, "hello")]), Tier("phones", 0, 1, [Interval(0, .5, "HH"), Interval(.5, .9, "AH")])])
    reject("cross word phone", [Tier("words", 0, 1, [Interval(0, .5, "hello"), Interval(.5, 1, "world")]),
                                 Tier("phones", 0, 1, [Interval(0, .49, "HH"), Interval(.49, .51, "AH"), Interval(.51, 1, "W")])],
           words=("hello", "world"))
    reject("phone gap", [Tier("words", 0, 1, [Interval(0, 1, "hello")]), Tier("phones", 0, 1, [Interval(0, .4, "HH"), Interval(.41, 1, "AH")])])
    reject("phone coverage", [Tier("words", 0, 1, [Interval(0, 1, "hello")]), Tier("phones", 0, 1, [Interval(.01, .5, "HH"), Interval(.5, 1, "AH")])])
    reject("extra outside phone", [Tier("words", 0, 1, [Interval(0, .8, "hello")]), Tier("phones", 0, 1, [Interval(0, .4, "HH"), Interval(.4, .8, "AH"), Interval(.8, 1, "W")])])

    ctc, aligned, output, segment = _ledger_fixture(root, "duplicate-ids", ("hello", "hello"))
    _tg(aligned / "x_seg0.TextGrid", [Tier("words", 0, 1, [Interval(0, .5, "HELLO"), Interval(.5, 1, "hello")]),
                                        Tier("phones", 0, 1, [Interval(0, .25, "HH"), Interval(.25, .5, "AH"),
                                                                Interval(.5, .75, "HH"), Interval(.75, 1, "AH")])])
    manifest = _ledger_manifest(ctc, aligned, output, segment)
    ledger = json.loads((output / "x_en_phones.json").read_text())
    words = ledger["segments"][0]["words"]
    fails += _expect(manifest["status"] == "success" and [word["word_id"] for word in words] == ["x:s0:w0", "x:s0:w1"]
                     and words[1]["phones"][0]["mfa_phone_ordinal"] == 2, "duplicate English word IDs and phone ordinals")

    ctc, aligned, output, segment = _ledger_fixture(root, "ctc-hash-missing")
    (ctc / "x.TextGrid").unlink()
    manifest = _ledger_manifest(ctc, aligned, output, segment)
    fails += _expect(manifest["rejected_segments"][0]["reason"].startswith("ctc_textgrid_hash_failed"), "CTC hash missing")

    ctc, aligned, output, segment = _ledger_fixture(root, "source-hash-failure")
    _tg(aligned / "x_seg0.TextGrid", valid)
    original_hash = align._sha256
    def source_hash_failure(path):
        if Path(path).name == "x_seg0.TextGrid":
            raise OSError("source hash fixture")
        return original_hash(path)
    with patch.object(align, "_sha256", side_effect=source_hash_failure):
        manifest = _ledger_manifest(ctc, aligned, output, segment)
    fails += _expect(manifest["rejected_segments"][0]["reason"].startswith("source_textgrid_hash_failed"), "source TG hash failure")

    ctc, aligned, output, segment = _ledger_fixture(root, "manifest")
    _tg(aligned / "x_seg0.TextGrid", valid)
    manifest = _ledger_manifest(ctc, aligned, output, segment)
    entry = manifest["stem_ledgers"][0]; ledger_path = Path(entry["path"])
    ids = set(manifest["produced_segments"]) | {item["id"] for item in manifest["rejected_segments"]}
    fails += _expect(manifest["status"] == "success" and ids == set(manifest["expected_segments"])
                     and not (set(manifest["produced_segments"]) & {item["id"] for item in manifest["rejected_segments"]})
                     and entry["sha256"] == align._sha256(ledger_path)
                     and manifest["counts"] == {"english_stems": 1, "english_segments": 1, "english_words": 1,
                                                "verified_words": 1, "rejected_words": 0},
                     "manifest exact partition counts and hash")
    inconsistent = json.loads(align.produce_strict_ledgers(segment, ctc, aligned, output, {"return_code": 0},
                                                            expected_segments=["wrong:s0"]).read_text())
    fails += _expect(inconsistent["status"] == "failed" and inconsistent["reason"] == "strict_manifest_inconsistent",
                     "manifest inconsistent denominator fails")
    return fails


def _postprocess_provenance_cases(root):
    """Matrix 17--26: strict consumer checks, including final written tier."""
    fails = 0

    def fixture(label, final_words=(Interval(2, 4, "hello"),), records=None,
                manifest_status="success", legacy=False):
        case = root / f"post-{label}"; en = case / "en"; en.mkdir(parents=True)
        source = case / "source.TextGrid"; source.write_text("source evidence", encoding="utf-8")
        if records is None:
            records = [{"word_id": "x:s0:w0", "ctc_ordinal": 0, "ctc_text": "hello",
                        "status": "verified", "provenance": "english_mfa_textgrid",
                        "mfa_word": {"ordinal": 0, "text": "hello", "start": 0.0, "end": 1.0},
                        "phones": [{"ordinal": 0, "mfa_phone_ordinal": 0, "label": "HH", "start": 0.0, "end": .25},
                                   {"ordinal": 1, "mfa_phone_ordinal": 1, "label": "AH", "start": .25, "end": 1.0}]}]
        ledger = {"schema": align.STRICT_SCHEMA, "stem": "x", "segments": [{
            "segment_id": "x:s0", "segment_ordinal": 0, "status": "verified",
            "mfa_textgrid": {"path": str(source), "sha256": post._strict_en_sha256(source)},
            "words": records}]}
        ledger_path = en / "x_en_phones.json"
        if legacy:
            ledger_path.write_text("[]", encoding="utf-8")
        else:
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
        expected = ["x:s0"] if manifest_status == "success" else []
        manifest = {"schema": align.STRICT_SCHEMA, "strict_provenance": True,
                    "status": manifest_status, "expected_segments": expected,
                    "produced_segments": expected,
                    "rejected_segments": [],
                    "stem_ledgers": [{"stem": "x", "path": str(ledger_path),
                                      "sha256": post._strict_en_sha256(ledger_path)}]}
        (en / "en_alignment_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return en, Tier("words", 0, 5, list(final_words))

    en, words = fixture("valid")
    report, pairs = post.load_strict_en_provenance("x", words, en)
    injected = post.inject_strict_en_phones(Tier("pinyin_phones", 0, 5,
                                                 [Interval(2, 4, "legacy")]), words, pairs)
    exact = [(iv.xmin, iv.xmax, iv.text) for iv in injected.intervals]
    fails += _expect(report["status"] == "verified" and exact ==
                     [(2.0, 2.5, "en:HH"), (2.5, 4.0, "en:AH")],
                     "strict exact sequence and affine mapping")

    en, words = fixture("legacy", legacy=True)
    fails += _expect(post.load_strict_en_provenance("x", words, en)[0]["status"] == "rejected",
                     "legacy list is rejected")
    (en / "x_en_phones.json").write_text(json.dumps([{
        "word_text": "hello", "word_start": 2.0, "word_end": 4.0,
        "en_word_start": 2.0, "en_word_end": 4.0,
        "phones": [{"phone": "HH", "start": 2.0, "end": 4.0}],
    }]), encoding="utf-8")
    legacy_data = post.load_en_phones("x", en)
    _, legacy_pp = post._apply_en_phones(words, Tier("pinyin_phones", 0, 5, []), legacy_data)
    fails += _expect(legacy_pp is not None and legacy_pp.intervals,
                     "non-strict legacy English path remains callable")
    bad = [{"word_id": "x:s0:w0", "ctc_ordinal": 0, "ctc_text": "hello", "status": "verified",
            "provenance": "english_mfa_textgrid", "mfa_word": {"ordinal": 0, "text": "hello", "start": 0, "end": 1}, "phones": []}]
    en, words = fixture("empty", records=bad)
    fails += _expect(post.load_strict_en_provenance("x", words, en)[0]["status"] == "rejected",
                     "empty phones are rejected")
    en, words = fixture("source")
    (en / "en_alignment_manifest.json").unlink()
    fails += _expect(post.load_strict_en_provenance("x", words, en)[0]["status"] == "rejected",
                     "missing manifest is rejected")
    en, words = fixture("hash")
    (en / "x_en_phones.json").write_text("{}", encoding="utf-8")
    fails += _expect(post.load_strict_en_provenance("x", words, en)[0]["status"] == "rejected",
                     "ledger hash mismatch is rejected")
    en, words = fixture("source-missing")
    source_path = next((root / "post-source-missing").glob("source.TextGrid"))
    source_path.unlink()
    fails += _expect(post.load_strict_en_provenance("x", words, en)[0]["status"] == "rejected",
                     "missing source evidence is rejected")
    rejected = [{"word_id": "x:s0:w0", "ctc_ordinal": 0, "ctc_text": "hello", "status": "rejected",
                 "provenance": None, "mfa_word": None, "phones": []}]
    en, words = fixture("rejected", records=rejected)
    # Segment status is the authoritative veto; a single rejected source word
    # cannot be rescued by any synthetic pronunciation.
    ledger_path = en / "x_en_phones.json"; ledger = json.loads(ledger_path.read_text())
    ledger["segments"][0]["status"] = "rejected"; ledger_path.write_text(json.dumps(ledger))
    manifest = json.loads((en / "en_alignment_manifest.json").read_text())
    manifest["stem_ledgers"][0]["sha256"] = post._strict_en_sha256(ledger_path)
    (en / "en_alignment_manifest.json").write_text(json.dumps(manifest))
    fails += _expect(post.load_strict_en_provenance("x", words, en)[0]["status"] == "rejected",
                     "one rejected word vetoes stem")
    cn_en, cn_words = fixture("chinese", final_words=(Interval(0, 1, "ni3"),), manifest_status="no_english")
    cn_report, _ = post.load_strict_en_provenance("x", cn_words, cn_en)
    fails += _expect(cn_report["status"] == "not_required", "pure Chinese is not required")

    # Repeated words and English separated by Chinese must bind by instance
    # ordering, not by a text+rounded-time dictionary key.
    records = [
        {"word_id": "x:s0:w0", "ctc_ordinal": 0, "ctc_text": "hello", "status": "verified",
         "provenance": "english_mfa_textgrid", "mfa_word": {"ordinal": 0, "text": "hello", "start": 0, "end": 1},
         "phones": [{"ordinal": 0, "mfa_phone_ordinal": 0, "label": "HH", "start": 0, "end": 1}]},
        {"word_id": "x:s0:w2", "ctc_ordinal": 2, "ctc_text": "hello", "status": "verified",
         "provenance": "english_mfa_textgrid", "mfa_word": {"ordinal": 1, "text": "hello", "start": 0, "end": 1},
         "phones": [{"ordinal": 0, "mfa_phone_ordinal": 1, "label": "AH", "start": 0, "end": 1}]},
    ]
    en, words = fixture("duplicates", final_words=(Interval(0, 1, "hello"), Interval(1, 2, "ni3"),
                                                    Interval(2, 3, "hello")), records=records)
    report, pairs = post.load_strict_en_provenance("x", words, en)
    injected = post.inject_strict_en_phones(Tier("pinyin_phones", 0, 3, []), words, pairs)
    fails += _expect(report["status"] == "verified" and [iv.text for iv in injected.intervals] == ["en:HH", "en:AH"],
                     "duplicate and Chinese-separated English instances")

    # End-to-end final output check: the legacy stress/boundary path runs
    # before strict injection, so a written strict tier must still preserve
    # the ledger's unstressed labels and exact affine edges.
    case = root / "post-process-final"; ctc = case / "input"; txt = case / "txt"; out = case / "out"; filt = case / "filt"
    ctc.mkdir(parents=True); txt.mkdir(); out.mkdir(); filt.mkdir()
    _tg(ctc / "x.TextGrid", [Tier("words", 0, 2, [Interval(0, 1, "hello"), Interval(1, 2, "ni3")]),
                               Tier("phones", 0, 2, [Interval(0, 1, "hello"), Interval(1, 1.5, "n"), Interval(1.5, 2, "i")])])
    (txt / "x.lab").write_text("hello ni3\n", encoding="utf-8")
    en, _ = fixture("final-evidence", final_words=(Interval(0, 1, "hello"),))
    args = SimpleNamespace(en_phones_dir=en, strict_ok=True, raw_text_dir=None, original_txt_dir=None,
        merge_silence=False, fix_short_word=False, enable_text_correction=False, handle_unexpected_sil=False,
        overwrite=True, filter_suspicious=True, filter_short_phone=False, detect_bgm=False,
        filter_edge_gap_sec=.25, filter_flank_silence_sec=.4, filter_long_word_sec=1.,
        filter_long_consonant_sec=999., filter_long_vowel_sec=999., filter_min_phone_coverage=.35,
        filter_min_word_dur_sec=.02, filter_min_word_sec=.15, filter_short_phone_sec=.015,
        filter_word_energy_ratio=2.)
    result = post.process_one(ctc / "x.TextGrid", txt, case / "wav", out, filt, args,
                              {"n": "n", "i": "i"}, {"ni3": ["n", "i3"]})
    written = post.parse_textgrid(Path(result["output"]))
    pp_tier = post.tier_by_name(written, "pinyin_phones")
    final_en = [(iv.xmin, iv.xmax, iv.text) for iv in pp_tier.intervals if iv.text.startswith("en:")]
    fails += _expect(result["english_provenance"]["status"] == "verified" and final_en ==
                     [(0.0, .25, "en:HH"), (.25, 1.0, "en:AH")],
                     "written strict output keeps exact affine labels and boundaries")

    missing_en = case / "missing-en"; missing_en.mkdir()
    rejected = post.process_one(ctc / "x.TextGrid", txt, case / "wav", out, filt,
                                SimpleNamespace(**{**vars(args), "en_phones_dir": missing_en}),
                                {"n": "n", "i": "i"}, {"ni3": ["n", "i3"]})
    rejected_tg = post.parse_textgrid(Path(rejected["output"]))
    rejected_pp = post.tier_by_name(rejected_tg, "pinyin_phones")
    fails += _expect(rejected["english_provenance"]["status"] == "rejected"
                     and "english_provenance_rejected" in rejected["filter_reasons"]
                     and not any(iv.text.startswith("en:") or iv.text == "hello"
                                 for iv in rejected_pp.intervals),
                     "filtered strict output has no English fallback phones")
    return fails


def main():
    fails = 0
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        c = root / "ledger-ctc"; a = root / "aligned"; o = root / "out"; c.mkdir(); a.mkdir(); o.mkdir()
        _tg(c / "x.TextGrid", [Tier("junk", 0, 2, [Interval(0, 2, "wrong")]),
                                Tier("words", 0, 2, [Interval(0, .5, "hello"), Interval(.5, 1, "ni3"), Interval(1, 1.5, "world")])])
        (c / "x.lab").write_text("hello ni3 world\n")
        seg = align.find_english_segments(c, ["x"])
        fails += _expect(len(seg["x"]) == 2 and seg["x"][0]["words"][0]["ordinal"] == 0 and seg["x"][1]["words"][0]["ordinal"] == 2,
                         "ordinal separators")
        seg["x"][0]["seg_name"] = "x_seg0"; seg["x"][1]["seg_name"] = "x_seg1"
        _tg(a / "x_seg0.TextGrid", [Tier("words", 0, 1, [Interval(0, .5, "hello")]), Tier("phones", 0, 1, [Interval(0, .2, "HH"), Interval(.2, .5, "AH")])])
        m = align.produce_strict_ledgers(seg, c, a, o, {"return_code": 0})
        d = json.loads(m.read_text()); ledger = json.loads((o / "x_en_phones.json").read_text())
        fails += _expect(d["schema"] == align.STRICT_SCHEMA and len(d["rejected_segments"]) == 1 and ledger["segments"][0]["words"][0]["word_id"] == "x:s0:w0",
                         "ledger schema and rejection")
        fails += _expect(ledger["segments"][0]["words"][0]["phones"][0]["label"] == "HH", "ARPABET provenance")
        expected, counts = align._strict_expected_snapshot(_segments())
        failed = align._strict_failed_manifest(o, mfa={"return_code": 4}, expected_segments=expected, expected_counts=counts)
        d = json.loads(failed.read_text())
        fails += _expect(d["expected_segments"] == expected and d["counts"]["english_segments"] == 1 and len(d["rejected_segments"]) == 1,
                         "failed manifest frozen expected denominator")
        fails += _g2p_failure_cases(root)
        fails += _corpus_cases(root)
        fails += _strict_mfa_runner_case(root)
        fails += _main_failure_cases(root)
        fails += _ledger_cases(root)
        fails += _postprocess_provenance_cases(root)
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
