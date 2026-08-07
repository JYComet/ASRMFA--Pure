#!/usr/bin/env python3
"""Filesystem-local acceptance checks for strict v4 CTC-ready import.

The suite uses a two-stem acoustic-rerun fixture.  It never reads or writes the
production run root and never starts MFA, ASR, or a GPU process.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pad_silence_edges as padding
import run_pipeline as runner


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)


def record(path: Path, *, wav: bool = False) -> dict:
    result = {"path": str(path.resolve()), "size": path.stat().st_size,
              "sha256": digest(path)}
    if wav:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            result["wav"] = {
                "frames": frames, "sample_rate": rate,
                "channels": handle.getnchannels(), "duration_s": frames / rate,
            }
    return result


class ReadyFixture:
    stems = ["000001_alpha", "000002_beta"]

    def __init__(self, base: Path):
        self.base = base
        self.authority = base / "authority"
        self.authority_audio = self.authority / "audio"
        self.authority_reference = self.authority / "reference"
        self.source_dictionary = self.authority / "dict" / "mfa_ipa.dict"
        self.run = base / "ready"
        self.audio = self.run / "audio_view"
        self.references = self.run / "reference_view"
        self.rerun = self.run / "ctc_rerun_output"
        self.ctc = self.run / "ctc_ready"
        self.dictionary = self.run / "dict" / "mfa_ipa.dict"
        for directory in (
                self.authority_audio, self.authority_reference,
                self.source_dictionary.parent, self.audio, self.references,
                self.rerun, self.ctc, self.dictionary.parent):
            directory.mkdir(parents=True, exist_ok=True)

        self.source_dictionary.write_text("hello\th ə l oʊ\n", encoding="utf-8")
        shutil.copy2(self.source_dictionary, self.dictionary)
        artifacts: dict[str, dict] = {}
        rerun_files: list[dict] = []
        for stem in self.stems:
            authoritative_wav = self.authority_audio / f"{stem}.wav"
            authoritative_ref = self.authority_reference / f"{stem}.txt"
            write_wav(authoritative_wav)
            authoritative_ref.write_text(f"reference {stem}\n", encoding="utf-8")
            wav = self.audio / authoritative_wav.name
            ref = self.references / authoritative_ref.name
            shutil.copy2(authoritative_wav, wav)
            shutil.copy2(authoritative_ref, ref)
            ctc: dict[str, dict] = {}
            for suffix in runner._CTC_SUFFIXES:
                source = self.rerun / f"{stem}{suffix}"
                destination = self.ctc / source.name
                source.write_text(f"{stem}:{suffix}\n", encoding="utf-8")
                shutil.copy2(source, destination)
                source_record = record(source)
                destination_record = record(destination)
                ctc[suffix] = destination_record
                rerun_files.append({
                    "kind": "rerun_ctc", "stem": stem,
                    "source": source_record, "destination": destination_record,
                })
            artifacts[stem] = {
                "origin_action": runner.STRICT_READY_ACTION,
                "audio": record(wav, wav=True),
                "reference": record(ref),
                "authoritative_audio": record(authoritative_wav, wav=True),
                "authoritative_reference": record(authoritative_ref),
                "ctc": ctc,
            }

        self.prepare_manifest = self.run / "prepare_manifest.json"
        self.prepare_manifest.write_text(json.dumps({
            "schema": runner.STRICT_READY_SCHEMA,
            "state": "awaiting_acoustic_rerun",
        }) + "\n", encoding="utf-8")
        self.taxonomy = [{
            "stem": stem,
            "reason": "legacy_audio_provenance_unbound",
            "action": runner.STRICT_READY_ACTION,
        } for stem in self.stems]
        self.taxonomy_sha256 = stable_digest(self.taxonomy)
        self.evidence = self.run / "ctc_ready_evidence.json"
        payload = {
            "schema": runner.STRICT_READY_SCHEMA,
            "state": "ready",
            "independent_verifier_signature": runner.STRICT_READY_VERIFIER_SIGNATURE,
            "prepare_manifest_sha256": digest(self.prepare_manifest),
            "inventory_sha256": stable_digest({"fixture": self.stems}),
            "authoritative_stems": self.stems,
            "stem_count": len(self.stems),
            "missing_reference": [],
            "txt_only": [],
            "final_audio_axis": runner.STRICT_READY_FINAL_AUDIO_AXIS,
            "padding_policy": runner.STRICT_READY_PADDING_POLICY,
            "action_counts": {runner.STRICT_READY_ACTION: len(self.stems)},
            "taxonomy": self.taxonomy,
            "taxonomy_sha256": self.taxonomy_sha256,
            "roots": {
                "run": str(self.run.resolve()),
                "audio_view": str(self.audio.resolve()),
                "reference_view": str(self.references.resolve()),
                "ctc_ready": str(self.ctc.resolve()),
            },
            "source_dictionary": record(self.source_dictionary),
            "run_local_dictionary": record(self.dictionary),
            "artifacts": artifacts,
            "rerun_files": rerun_files,
            "rerun_files_sha256": stable_digest(rerun_files),
        }
        self.evidence.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    def config(self, workspace: Path) -> dict:
        return {
            "mode": "ctc_ready", "workspace": str(workspace),
            "data_dir": str(self.audio), "mfa_dict": str(self.dictionary),
            "runtime_mfa_dict": "runtime/mfa_ipa.dict",
            "disable_nvme_cache": True, "nvme_cache": "",
            "output_staging": False, "use_cache": False,
            "pad_silence": {"enabled": False},
            "ctc_adjust": {"enabled": True, "limit": 0},
            "postprocess": {"strict_ok": True},
            "mfa": {"num_jobs": 1},
            "mfa_en": {"enabled": True, "strict_provenance": True},
            "ctc_ready": {
                "authoritative_source_dir": str(self.authority),
                "source_dictionary": str(self.source_dictionary),
                "ctc_dir": str(self.ctc), "text_dir": str(self.references),
                "require_all": True, "isolate_copy": True,
                "expected_count": len(self.stems),
                "require_fresh_workspace": True, "verify_timeout": 30,
                "expected_ready_evidence": {
                    "path": str(self.evidence), "sha256": digest(self.evidence),
                    "taxonomy_sha256": self.taxonomy_sha256,
                    "schema": runner.STRICT_READY_SCHEMA, "state": "ready",
                    "independent_verifier_signature":
                        runner.STRICT_READY_VERIFIER_SIGNATURE,
                },
            },
        }

    def context(self, workspace: Path) -> dict:
        return {
            "workspace": workspace,
            "ctc_pretg": workspace / "ctc_pretg",
            "ctc_pretg_adj": workspace / "ctc_pretg_adj",
            "data_dir": self.audio,
            "audio_dir": self.audio,
            "mfa_audio_dir": workspace / "audio_16k",
            "mfa_dict": self.dictionary,
            "strict_ready": True,
        }


def small_contract(source_root: Path):
    return mock.patch.multiple(
        runner,
        STRICT_READY_COUNT=2,
        STRICT_READY_MISSING_REFERENCES=[],
        STRICT_READY_AUTHORITATIVE_SOURCE=source_root,
        STRICT_READY_SOURCE_DICTIONARY=source_root / "dict" / "mfa_ipa.dict",
    )


class StrictImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="strict-ctc-import-")
        self.base = Path(self.temp.name)
        self.fixture = ReadyFixture(self.base)
        self.verifier_hook = mock.Mock(return_value=0)

    def tearDown(self):
        self.temp.cleanup()

    def _import(self, selector: Path | None = None,
                name: str = "workspace") -> tuple[int, dict]:
        workspace = self.base / name
        workspace.mkdir()
        cfg = self.fixture.config(workspace)
        ctx = self.fixture.context(workspace)
        args = SimpleNamespace(
            ctc_ready_stems_file=str(selector) if selector else None)
        with small_contract(self.fixture.authority), mock.patch.object(
                runner, "STRICT_READY_VERIFY_HOOK", self.verifier_hook):
            rc = runner._step_link_ctc_strict(args, cfg, ctx)
        return rc, ctx

    def test_full_import_is_exact_not_aliased_and_uses_verifier_hook(self):
        rc, ctx = self._import()
        self.assertEqual(rc, 0)
        self.assertEqual(2, self.verifier_hook.call_count)
        self.assertEqual(tuple(self.fixture.stems), ctx["expected_stems"])
        self.assertEqual(self.fixture.audio.resolve(), ctx["audio_dir"])
        expected_names = {
            f"{stem}{suffix}" for stem in self.fixture.stems
            for suffix in runner._CTC_SUFFIXES}
        expected_names |= {f"{stem}_ref.txt" for stem in self.fixture.stems}
        self.assertEqual(
            expected_names, {path.name for path in ctx["ctc_pretg"].iterdir()})
        source = self.fixture.ctc / f"{self.fixture.stems[0]}.lab"
        destination = ctx["ctc_pretg"] / source.name
        self.assertEqual(digest(source), digest(destination))
        self.assertNotEqual(
            (source.stat().st_dev, source.stat().st_ino),
            (destination.stat().st_dev, destination.stat().st_ino))
        manifest = json.loads(
            (ctx["workspace"] / "ctc_ready_import_manifest.json").read_text(
                encoding="utf-8"))
        self.assertEqual({runner.STRICT_READY_ACTION: 2},
                         manifest["evidence_action_counts"])
        self.assertEqual(self.fixture.taxonomy_sha256,
                         manifest["evidence_taxonomy_sha256"])
        self.assertEqual(2, manifest["full_evidence_count"])
        self.assertEqual(2, manifest["selected_count"])
        self.assertTrue(manifest["checks"]["source_verify_after"])
        for step in ("link", "normalize_punct", "normalize", "normalize_ria",
                     "normalize_en"):
            self.assertEqual([], runner.strict_stage_denominator_issues(step, ctx))
        self.assertFalse((ctx["workspace"] / "padded_audio").exists())
        source_dict_hash = digest(self.fixture.dictionary)
        Path(ctx["mfa_dict"]).write_text("runtime-only\n", encoding="utf-8")
        self.assertEqual(source_dict_hash, digest(self.fixture.dictionary))

    def test_canary_selector_and_selector_rejections(self):
        selector = self.base / "selector.txt"
        selector.write_text(self.fixture.stems[1] + "\n", encoding="utf-8")
        rc, ctx = self._import(selector, "canary")
        self.assertEqual(rc, 0)
        self.assertEqual((self.fixture.stems[1],), ctx["expected_stems"])
        self.assertEqual(2, len(list(ctx["audio_dir"].glob("*.wav"))))
        duplicate = self.base / "duplicate.txt"
        duplicate.write_text(
            self.fixture.stems[0] + "\n" + self.fixture.stems[0] + "\n",
            encoding="utf-8")
        unknown = self.base / "unknown.txt"
        unknown.write_text("999999_unknown\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            runner.load_strict_stem_selection(duplicate, self.fixture.stems)
        with self.assertRaises(ValueError):
            runner.load_strict_stem_selection(unknown, self.fixture.stems)

    def test_canary_resample_consumes_only_selected_audio(self):
        selector = self.base / "selector-resample.txt"
        selector.write_text(self.fixture.stems[1] + "\n", encoding="utf-8")
        rc, ctx = self._import(selector, "canary-resample")
        self.assertEqual(rc, 0)
        consumed: list[str] = []

        def fake_resample(wav_path, _audio_dir, out_dir, _target_sr, _overwrite):
            consumed.append(wav_path.name)
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(wav_path, out_dir / wav_path.name)
            return wav_path.name, True, "copied"

        with mock.patch.object(runner, "_resample_one", side_effect=fake_resample):
            result = runner.step_resample_for_mfa(
                SimpleNamespace(overwrite=False),
                self.fixture.config(ctx["workspace"]), Path(os.sys.executable), ctx)
        self.assertEqual(0, result)
        self.assertEqual([f"{self.fixture.stems[1]}.wav"], consumed)
        self.assertEqual(
            {f"{self.fixture.stems[1]}.wav"},
            {path.name for path in ctx["mfa_audio_dir"].iterdir()})
        self.assertEqual([], runner.strict_stage_denominator_issues("resample", ctx))

    def test_copy_detects_source_mutation_and_alias(self):
        source = self.base / "source"
        source.write_bytes(b"before")
        destination = self.base / "destination"
        evidence = record(source)
        original_copy = shutil.copyfile

        def mutating_copy(src, dst):
            result = original_copy(src, dst)
            Path(src).write_bytes(b"after!")
            return result

        with mock.patch("shutil.copyfile", side_effect=mutating_copy):
            with self.assertRaises(ValueError):
                runner._copy_regular_verified(source, destination, evidence)
        alias_source = self.base / "alias-source"
        alias_source.write_bytes(b"same")
        alias_destination = self.base / "alias-destination"
        os.link(alias_source, alias_destination)
        stat = alias_source.stat()
        alias_record = {
            "source": str(alias_source), "destination": str(alias_destination),
            "size": stat.st_size, "sha256": digest(alias_source),
            "source_dev": stat.st_dev, "source_ino": stat.st_ino,
            "destination_dev": stat.st_dev, "destination_ino": stat.st_ino,
        }
        with self.assertRaises(ValueError):
            runner._verify_imported_copy(alias_record, self.base)

    def test_evidence_contract_and_tamper_rejections(self):
        baseline = json.loads(self.fixture.evidence.read_text(encoding="utf-8"))

        def assert_rejected(payload: dict, extra: Path | None = None):
            self.fixture.evidence.write_text(
                json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            cfg = self.fixture.config(self.base / "unused-workspace")
            if extra is not None:
                extra.write_text("extra\n", encoding="utf-8")
            try:
                with small_contract(self.fixture.authority), self.assertRaises(ValueError):
                    runner.load_and_validate_ready_evidence(cfg)
            finally:
                if extra is not None:
                    extra.unlink(missing_ok=True)

        variants: list[dict] = []
        payload = copy.deepcopy(baseline)
        payload["schema"] = "hecheng-english-ctc-ready-v3"
        variants.append(payload)
        payload = copy.deepcopy(baseline)
        payload["state"] = "stale"
        variants.append(payload)
        payload = copy.deepcopy(baseline)
        payload["stem_count"] = 1
        variants.append(payload)
        payload = copy.deepcopy(baseline)
        payload["authoritative_stems"] = list(reversed(payload["authoritative_stems"]))
        variants.append(payload)
        payload = copy.deepcopy(baseline)
        payload["action_counts"] = {runner.STRICT_READY_ACTION: 1}
        variants.append(payload)
        payload = copy.deepcopy(baseline)
        payload["final_audio_axis"] = "padded_audio"
        variants.append(payload)
        payload = copy.deepcopy(baseline)
        payload["padding_policy"] = "allowed"
        variants.append(payload)
        payload = copy.deepcopy(baseline)
        payload["roots"]["audio_view"] = str(self.base / "redirected-audio")
        variants.append(payload)
        payload = copy.deepcopy(baseline)
        payload["missing_reference"] = ["unexpected_missing"]
        variants.append(payload)
        payload = copy.deepcopy(baseline)
        payload["independent_verifier_signature"] = "wrong-verifier"
        variants.append(payload)
        payload = copy.deepcopy(baseline)
        payload["taxonomy"][0]["reason"] = "tampered"
        variants.append(payload)
        payload = copy.deepcopy(baseline)
        payload["taxonomy_sha256"] = "0" * 64
        variants.append(payload)
        payload = copy.deepcopy(baseline)
        payload["artifacts"][self.fixture.stems[0]]["origin_action"] = "legacy_copy"
        variants.append(payload)
        payload = copy.deepcopy(baseline)
        payload["artifacts"][self.fixture.stems[0]]["authoritative_audio"]["sha256"] = "0" * 64
        variants.append(payload)
        payload = copy.deepcopy(baseline)
        payload["artifacts"][self.fixture.stems[0]]["reference"]["path"] = str(
            self.fixture.references / f"{self.fixture.stems[1]}.txt")
        variants.append(payload)
        payload = copy.deepcopy(baseline)
        del payload["artifacts"][self.fixture.stems[0]]["ctc"][
            runner._CTC_SUFFIXES[0]]
        variants.append(payload)
        payload = copy.deepcopy(baseline)
        payload["unexpected_v3_field"] = {}
        variants.append(payload)
        for payload in variants:
            with self.subTest(schema=payload.get("schema"),
                              axis=payload.get("final_audio_axis"),
                              padding=payload.get("padding_policy")):
                assert_rejected(payload)
        assert_rejected(copy.deepcopy(baseline), self.fixture.ctc / "unexpected.file")

        self.fixture.evidence.write_text(
            json.dumps(baseline, ensure_ascii=False) + "\n", encoding="utf-8")
        cfg = self.fixture.config(self.base / "unused-workspace")
        cfg["ctc_ready"]["expected_ready_evidence"]["sha256"] = "0" * 64
        with small_contract(self.fixture.authority), self.assertRaises(ValueError):
            runner.load_and_validate_ready_evidence(cfg)

    def test_preflight_rejects_nonfresh_placeholders_v3_and_padding(self):
        workspace = self.base / "future-workspace"
        cfg = self.fixture.config(workspace)
        attrs = {
            "force": False, "overwrite": False, "use_cache": False,
            "auto_cache": False, "scan_only": False, "output_staging": False,
            "no_output_staging": True, "step": None, "skip_to": None,
            "stop_after": None, "ctc_ready": None, "data_dir": None,
            "nvme_cache": None, "output_dir": None, "cache_dir": None,
            "dataset_offset": 0, "dataset_limit": 0, "workspace": str(workspace),
        }
        attrs.update({f"skip_{name}": False for name in runner.STEPS})
        args = SimpleNamespace(**attrs)
        with small_contract(self.fixture.authority):
            self.assertEqual(
                workspace,
                runner.validate_strict_ready_invocation(
                    args, cfg, "ctc_ready", False))
            workspace.mkdir()
            with self.assertRaises(ValueError):
                runner.validate_strict_ready_invocation(
                    args, cfg, "ctc_ready", False)
            workspace.rmdir()

            for mutate in ("evidence", "taxonomy", "v3", "padding"):
                bad = copy.deepcopy(cfg)
                if mutate == "evidence":
                    bad["ctc_ready"]["expected_ready_evidence"]["sha256"] = \
                        "REPLACE_AFTER_FINALIZE"
                elif mutate == "taxonomy":
                    bad["ctc_ready"]["expected_ready_evidence"][
                        "taxonomy_sha256"] = "REPLACE_AFTER_FINALIZE"
                elif mutate == "v3":
                    bad["ctc_ready"]["expected_ready_evidence"]["schema"] = \
                        "hecheng-english-ctc-ready-v3"
                else:
                    bad["pad_silence"]["enabled"] = True
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    runner.validate_strict_ready_invocation(
                        args, bad, "ctc_ready", False)

        real_parent = self.base / "real-parent"
        real_parent.mkdir()
        linked_parent = self.base / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        linked_workspace = linked_parent / "workspace"
        cfg = self.fixture.config(linked_workspace)
        args.workspace = str(linked_workspace)
        with small_contract(self.fixture.authority), self.assertRaises(ValueError):
            runner.validate_strict_ready_invocation(args, cfg, "ctc_ready", False)

    def test_strict_route_has_no_padding_stage(self):
        self.assertNotIn("pad_silence", runner.STRICT_CTC_READY_STEP_ORDER)
        self.assertEqual("link", runner.STRICT_CTC_READY_STEP_ORDER[0])
        self.assertEqual("strict_ok", runner.STRICT_CTC_READY_STEP_ORDER[-1])
        self.assertEqual(len(runner.STRICT_CTC_READY_STEP_ORDER),
                         len(set(runner.STRICT_CTC_READY_STEP_ORDER)))

    def test_independent_verifier_command_is_direct(self):
        cfg = self.fixture.config(self.base / "unused-workspace")
        with small_contract(self.fixture.authority):
            evidence = runner.load_and_validate_ready_evidence(cfg)
        completed = SimpleNamespace(returncode=0)
        with mock.patch.object(runner.subprocess, "run", return_value=completed) as invoke:
            self.assertEqual(0, runner._run_ready_verifier(cfg, evidence))
        command = invoke.call_args.args[0]
        self.assertEqual(os.sys.executable, command[0])
        self.assertEqual(
            str(runner.SCRIPTS_DIR / "verify_hecheng_english_ctc_ready_v4.py"),
            command[1])
        self.assertEqual([
            "--run-root", str(self.fixture.run.resolve()),
            "--source-dir", str(self.fixture.authority.resolve()),
            "--dictionary-source", str(self.fixture.source_dictionary.resolve()),
        ], command[2:])
        self.assertNotIn("prepare_hecheng_english_ctc_ready.py", " ".join(command))

    def test_active_audio_denominator_shrink_fails(self):
        rc, ctx = self._import(name="denominator")
        self.assertEqual(rc, 0)
        self.assertEqual([], runner.strict_stage_denominator_issues("link", ctx))
        (self.fixture.audio / f"{self.fixture.stems[0]}.wav").unlink()
        self.assertTrue(runner.strict_stage_denominator_issues("link", ctx))


class ProductionConfigTests(unittest.TestCase):
    def test_english_config_is_blocked_v4_without_padding_or_publication(self):
        cfg = runner.load_config(
            runner.PROJECT_ROOT / "configs" / "hecheng_english_mfa.yaml")
        root = Path(
            "/mnt/nvme3/mfa_runs/hecheng_english/20260806_strict_v4_0")
        ready = cfg["ctc_ready"]
        pin = ready["expected_ready_evidence"]
        self.assertEqual("ctc_ready", cfg["mode"])
        self.assertEqual(str(root / "audio_view"), cfg["data_dir"])
        self.assertEqual(str(root / "workspace_full"), cfg["workspace"])
        self.assertEqual(str(root / "ctc_ready"), ready["ctc_dir"])
        self.assertEqual(str(root / "reference_view"), ready["text_dir"])
        self.assertEqual(str(runner.STRICT_READY_AUTHORITATIVE_SOURCE),
                         ready["authoritative_source_dir"])
        self.assertEqual(str(runner.STRICT_READY_SOURCE_DICTIONARY),
                         ready["source_dictionary"])
        self.assertIs(cfg["pad_silence"]["enabled"], False)
        self.assertIs(cfg["output_staging"], False)
        self.assertEqual(runner.STRICT_READY_SCHEMA, pin["schema"])
        self.assertEqual(runner.STRICT_READY_VERIFIER_SIGNATURE,
                         pin["independent_verifier_signature"])
        self.assertEqual("REPLACE_AFTER_FINALIZE", pin["sha256"])
        self.assertEqual("REPLACE_AFTER_FINALIZE", pin["taxonomy_sha256"])


class PaddingContractTests(unittest.TestCase):
    """Legacy/non-v4 padding remains fail-closed even though v4 forbids it."""

    def test_worker_exception_missing_and_extra_outputs_fail(self):
        with tempfile.TemporaryDirectory(prefix="strict-padding-") as raw:
            padded = Path(raw)
            for stem in ("a", "b"):
                (padded / f"{stem}.wav").write_bytes(b"wav")
            success = [{"stem": "a"}, {"stem": "b"}]
            self.assertEqual(
                [], padding.validate_completion(["a", "b"], success, padded, False))
            worker_error = [{"stem": "a"}, {"stem": "b", "error": "boom"}]
            self.assertTrue(padding.validate_completion(
                ["a", "b"], worker_error, padded, False))
            (padded / "b.wav").unlink()
            self.assertTrue(padding.validate_completion(
                ["a", "b"], success, padded, False))
            (padded / "b.wav").write_bytes(b"wav")
            (padded / "extra.wav").write_bytes(b"wav")
            self.assertTrue(padding.validate_completion(
                ["a", "b"], success, padded, False))

    def test_padding_main_returns_nonzero_for_worker_exception_or_missing_wav(self):
        with tempfile.TemporaryDirectory(prefix="strict-padding-main-") as raw:
            root = Path(raw)
            ctc = root / "ctc"
            audio = root / "audio"
            ctc.mkdir()
            audio.mkdir()
            stems_file = root / "stems.txt"
            stems_file.write_text("a\nb\n", encoding="utf-8")
            for stem in ("a", "b"):
                (ctc / f"{stem}.lab").write_text(stem, encoding="utf-8")
            argv = [
                "pad_silence_edges.py", "--ctc-dir", str(ctc),
                "--audio-dir", str(audio),
                "--padded-audio-dir", str(root / "padded"),
                "--stems-file", str(stems_file),
            ]
            with mock.patch.object(os.sys, "argv", argv), mock.patch.object(
                    padding, "process_one", side_effect=RuntimeError("boom")):
                self.assertNotEqual(0, padding.main())

            def no_output(stem, *_args, **_kwargs):
                return {"stem": stem}

            with mock.patch.object(os.sys, "argv", argv), mock.patch.object(
                    padding, "process_one", side_effect=no_output):
                self.assertNotEqual(0, padding.main())


# ═══════════════════════════════════════════════════════════════════════
# Model tree + receipt tests (Case 99 / R5)
# ═══════════════════════════════════════════════════════════════════════

class ModelTreeReceiptTests(unittest.TestCase):
    """Fault tests for model tree digest, CTC run receipt, and shard receipt."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write_model_tree(self, base: Path, files: dict[str, str]) -> None:
        """Write a model tree fixture: {relpath: content}."""
        base.mkdir(parents=True, exist_ok=True)
        for rel, content in files.items():
            p = base / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

    def test_model_tree_digest_is_deterministic(self):
        """Same file tree produces same digest across two calls."""
        from pipeline_utils import compute_model_tree_digest
        base = self.root / "model_v1"
        self._write_model_tree(base, {"model.pt": "weights", "config.json": '{"layers": 12}'})
        d1, m1 = compute_model_tree_digest(base)
        d2, m2 = compute_model_tree_digest(base)
        self.assertEqual(d1, d2)
        self.assertEqual(m1, m2)

    def test_model_tree_digest_rejects_symlink(self):
        """A symlink in the model tree raises ValueError."""
        from pipeline_utils import compute_model_tree_digest
        base = self.root / "model_sym"
        self._write_model_tree(base, {"model.pt": "weights"})
        # Create a symlink inside the tree
        sym = base / "link.pt"
        os.symlink(str(base / "model.pt"), str(sym))
        with self.assertRaises(ValueError):
            compute_model_tree_digest(base)

    def test_model_tree_digest_detects_content_change(self):
        """Replacing a file changes the tree digest."""
        from pipeline_utils import compute_model_tree_digest
        base = self.root / "model_chg"
        self._write_model_tree(base, {"model.pt": "weights_v1"})
        d1, _ = compute_model_tree_digest(base)
        # Replace content
        (base / "model.pt").write_text("weights_v2", encoding="utf-8")
        d2, _ = compute_model_tree_digest(base)
        self.assertNotEqual(d1, d2)

    def test_model_tree_digest_detects_file_addition(self):
        """Adding a file changes the tree digest."""
        from pipeline_utils import compute_model_tree_digest
        base = self.root / "model_add"
        self._write_model_tree(base, {"model.pt": "weights"})
        d1, _ = compute_model_tree_digest(base)
        # Add a file
        (base / "tokenizer.json").write_text("{}", encoding="utf-8")
        d2, _ = compute_model_tree_digest(base)
        self.assertNotEqual(d1, d2)

    def test_model_tree_digest_manifest_matches_files(self):
        """File manifest entries match actual file sizes and hashes."""
        from pipeline_utils import compute_model_tree_digest, _sha256_file
        base = self.root / "model_manifest"
        self._write_model_tree(base, {"model.pt": "weights", "config.json": "{}"})
        _, manifest = compute_model_tree_digest(base)
        self.assertEqual(len(manifest), 2)
        for entry in manifest:
            p = base / entry["relpath"]
            self.assertEqual(entry["size"], p.stat().st_size)
            self.assertEqual(entry["sha256"], _sha256_file(p))

    def test_run_receipt_is_atomic_and_binds_all_fields(self):
        """Write receipt, read back, verify all keys present and digests match."""
        from pipeline_utils import (compute_model_tree_digest, write_ctc_run_receipt)
        model_dir = self.root / "model_rec"
        self._write_model_tree(model_dir, {"model.pt": "weights"})
        tree_digest, manifest = compute_model_tree_digest(model_dir)
        dict_dir = self.root / "dict"
        dict_dir.mkdir()
        dict_path = dict_dir / "mfa_ipa.dict"
        dict_path.write_text("a a\n", encoding="utf-8")
        dict_dig = hashlib.sha256(dict_path.read_bytes()).hexdigest()

        out = self.root / "output"
        out.mkdir()
        receipt = write_ctc_run_receipt(
            out, actual_argv=["python", "ctc_prealign.py"],
            asr_python="/usr/bin/python",
            model_path=model_dir,
            model_tree_digest=tree_digest,
            model_file_manifest=manifest,
            dict_path=dict_path,
            dict_digest=dict_dig,
            input_stems=["s1", "s2"],
            output_stems=["s1", "s2"],
        )
        self.assertEqual(receipt["schema"], "ctc-run-receipt-v1")
        self.assertEqual(receipt["model"]["tree_digest"], tree_digest)
        self.assertEqual(receipt["input_stems"], ["s1", "s2"])
        # Verify atomic write
        receipt_file = out / ".ctc_run_receipt.json"
        self.assertTrue(receipt_file.is_file())
        reloaded = json.loads(receipt_file.read_text(encoding="utf-8"))
        self.assertEqual(reloaded["model"]["tree_digest"], tree_digest)

    def test_run_receipt_mismatched_model_digest_detected(self):
        """Modifying receipt's model_tree_digest causes cross-check to fail."""
        from pipeline_utils import (compute_model_tree_digest, write_ctc_run_receipt)
        model_dir = self.root / "model_mm"
        self._write_model_tree(model_dir, {"model.pt": "weights"})
        tree_digest, manifest = compute_model_tree_digest(model_dir)
        dict_path = self.root / "dict_mm" / "dict.txt"
        dict_path.parent.mkdir()
        dict_path.write_text("a a\n", encoding="utf-8")
        dict_dig = hashlib.sha256(dict_path.read_bytes()).hexdigest()

        out = self.root / "output_mm"
        out.mkdir()
        write_ctc_run_receipt(out, actual_argv=["p"], asr_python="/usr/bin/p",
                              model_path=model_dir, model_tree_digest=tree_digest,
                              model_file_manifest=manifest, dict_path=dict_path,
                              dict_digest=dict_dig, input_stems=["s1"], output_stems=["s1"])
        # Tamper the receipt
        receipt_file = out / ".ctc_run_receipt.json"
        data = json.loads(receipt_file.read_text(encoding="utf-8"))
        data["model"]["tree_digest"] = "deadbeef"
        receipt_file.write_text(json.dumps(data), encoding="utf-8")
        # Re-read and check
        tampered = json.loads(receipt_file.read_text(encoding="utf-8"))
        self.assertNotEqual(tampered["model"]["tree_digest"], tree_digest)


class MfaTextGridValidatorTests(unittest.TestCase):
    """Fault tests for strict MFA TextGrid validator (Cases 76/83 / R7)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write_tg(self, path: Path, xmin: float = 0.0, xmax: float = 2.0,
                  tiers: list[dict] | None = None) -> None:
        """Write a minimal valid long-format TextGrid."""
        if tiers is None:
            tiers = [
                {"name": "words", "xmin": 0.0, "xmax": 2.0,
                 "intervals": [(0.0, 1.0, "hello"), (1.0, 2.0, "world")]},
                {"name": "phones", "xmin": 0.0, "xmax": 2.0,
                 "intervals": [(0.0, 0.5, "hh"), (0.5, 1.0, "ow"), (1.0, 1.5, "w"), (1.5, 2.0, "d")]},
            ]
        lines = [
            'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
            f"xmin = {xmin} ", f"xmax = {xmax} ",
            "tiers? <exists> ", f"size = {len(tiers)} ", "item []: ",
        ]
        for ti, tier in enumerate(tiers, start=1):
            lines.extend([
                f"    item [{ti}]:", '        class = "IntervalTier" ',
                f'        name = "{tier["name"]}" ',
                f"        xmin = {tier['xmin']} ", f"        xmax = {tier['xmax']} ",
                f"        intervals: size = {len(tier['intervals'])} ",
            ])
            for ji, (ix, iy, it) in enumerate(tier["intervals"], start=1):
                lines.extend([
                    f"        intervals [{ji}]:",
                    f"            xmin = {ix} ",
                    f"            xmax = {iy} ",
                    f'            text = "{it}" ',
                ])
        path.write_text("\n".join(lines), encoding="utf-8")

    def test_valid_textgrid_passes(self):
        """A well-formed words+phones TextGrid returns empty error list."""
        from pipeline_utils import validate_strict_mfa_textgrid
        tg = self.root / "valid.TextGrid"
        self._write_tg(tg)
        errors = validate_strict_mfa_textgrid(tg)
        self.assertEqual(errors, [])

    def test_corrupt_header_fails(self):
        """Missing File type header produces an error."""
        from pipeline_utils import validate_strict_mfa_textgrid
        tg = self.root / "bad_header.TextGrid"
        tg.write_text("not a TextGrid file\n", encoding="utf-8")
        errors = validate_strict_mfa_textgrid(tg)
        self.assertTrue(len(errors) > 0)

    def test_tier_name_duplicate_fails(self):
        """Two tiers named 'words' produces an error."""
        from pipeline_utils import validate_strict_mfa_textgrid
        tg = self.root / "dup_tier.TextGrid"
        self._write_tg(tg, tiers=[
            {"name": "words", "xmin": 0.0, "xmax": 1.0,
             "intervals": [(0.0, 1.0, "hi")]},
            {"name": "words", "xmin": 0.0, "xmax": 1.0,
             "intervals": [(0.0, 1.0, "h")]},
        ])
        errors = validate_strict_mfa_textgrid(tg)
        dup_errors = [e for e in errors if "duplicate" in e]
        self.assertTrue(len(dup_errors) > 0, f"Expected duplicate tier error, got: {errors}")

    def test_inverted_interval_fails(self):
        """Interval with xmin > xmax produces an error."""
        from pipeline_utils import validate_strict_mfa_textgrid
        tg = self.root / "inverted.TextGrid"
        self._write_tg(tg, tiers=[
            {"name": "words", "xmin": 0.0, "xmax": 2.0,
             "intervals": [(1.0, 0.5, "bad")]},
            {"name": "phones", "xmin": 0.0, "xmax": 2.0,
             "intervals": [(0.0, 1.0, "h")]},
        ])
        errors = validate_strict_mfa_textgrid(tg)
        self.assertTrue(len(errors) > 0)

    def test_zero_duration_interval_fails(self):
        """Interval with xmin == xmax produces an error."""
        from pipeline_utils import validate_strict_mfa_textgrid
        tg = self.root / "zero_dur.TextGrid"
        self._write_tg(tg, tiers=[
            {"name": "words", "xmin": 0.0, "xmax": 2.0,
             "intervals": [(0.5, 0.5, "bad")]},
            {"name": "phones", "xmin": 0.0, "xmax": 2.0,
             "intervals": [(0.0, 1.0, "h")]},
        ])
        errors = validate_strict_mfa_textgrid(tg)
        self.assertTrue(len(errors) > 0)

    def test_non_monotonic_intervals_fail(self):
        """Interval that starts before the previous one ends fails."""
        from pipeline_utils import validate_strict_mfa_textgrid
        tg = self.root / "nonmono.TextGrid"
        self._write_tg(tg, tiers=[
            {"name": "words", "xmin": 0.0, "xmax": 3.0,
             "intervals": [(0.0, 1.0, "first"), (0.5, 2.0, "overlap")]},
            {"name": "phones", "xmin": 0.0, "xmax": 3.0,
             "intervals": [(0.0, 0.5, "f"), (0.5, 1.0, "o")]},
        ])
        errors = validate_strict_mfa_textgrid(tg)
        self.assertTrue(len(errors) > 0)

    def test_textgrid_exceeds_wav_domain_fails(self):
        """TextGrid with xmax > wav_duration_s produces an error."""
        from pipeline_utils import validate_strict_mfa_textgrid
        tg = self.root / "overdomain.TextGrid"
        self._write_tg(tg, xmax=10.0, tiers=[
            {"name": "words", "xmin": 0.0, "xmax": 10.0,
             "intervals": [(0.0, 1.0, "hi")]},
            {"name": "phones", "xmin": 0.0, "xmax": 10.0,
             "intervals": [(0.0, 1.0, "h")]},
        ])
        errors = validate_strict_mfa_textgrid(tg, wav_duration_s=3.0)
        domain_errors = [e for e in errors if "exceeds WAV" in e]
        self.assertTrue(len(domain_errors) > 0, f"Expected domain error, got: {errors}")

    def test_missing_phones_tier_fails(self):
        """TextGrid with only words tier is rejected."""
        from pipeline_utils import validate_strict_mfa_textgrid
        tg = self.root / "no_phones.TextGrid"
        self._write_tg(tg, tiers=[
            {"name": "words", "xmin": 0.0, "xmax": 2.0,
             "intervals": [(0.0, 2.0, "hi")]},
        ])
        errors = validate_strict_mfa_textgrid(tg)
        missing = [e for e in errors if "phones" in e]
        self.assertTrue(len(missing) > 0, f"Expected missing phones error, got: {errors}")

    def test_string_match_would_pass_but_parser_fails(self):
        """TextGrid containing name='words' as text in an interval but no actual tier."""
        from pipeline_utils import validate_strict_mfa_textgrid
        tg = self.root / "sneaky.TextGrid"
        lines = [
            'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
            "xmin = 0.0 ", "xmax = 2.0 ",
            "tiers? <exists> ", "size = 1 ", "item []: ",
            "    item [1]:", '        class = "IntervalTier" ',
            '        name = "other" ',
            "        xmin = 0.0 ", "        xmax = 2.0 ",
            "        intervals: size = 2 ",
            "        intervals [1]:",
            "            xmin = 0.0 ",
            "            xmax = 1.0 ",
            '            text = "name = \\"words\\" name = \\"phones\\"" ',
            "        intervals [2]:",
            "            xmin = 1.0 ",
            "            xmax = 2.0 ",
            '            text = "" ',
        ]
        tg.write_text("\n".join(lines), encoding="utf-8")
        errors = validate_strict_mfa_textgrid(tg)
        # Must have missing words/phones tier, not pass
        self.assertTrue(len(errors) > 0, f"Should have failed but got no errors")
        has_missing = any("words" in e or "phones" in e for e in errors)
        self.assertTrue(has_missing, f"Should report missing words/phones tier, got: {errors}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(
        verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
