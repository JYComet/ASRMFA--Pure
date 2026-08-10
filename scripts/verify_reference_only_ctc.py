#!/usr/bin/env python3
"""No-GPU fault tests for the English reference-only CTC contract."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import ctc_prealign as ctc
import verify_hecheng_english_ctc_ready_v4 as verifier
import prepare_hecheng_english_ctc_ready as prepare


def write_bundle(root: Path, stem: str, reference: str, *, extra_nvv: bool = False,
                 reorder_punct: bool = False) -> None:
    lexical = ["ni3", "hao3", "BREATHING"]
    if extra_nvv:
        lexical.insert(1, "LAUGHTER")
    (root / f"{stem}.lab").write_text(" ".join(lexical) + "\n", encoding="utf-8")
    rows = []
    for index, word in enumerate(lexical):
        rows.append({"word": word, "start_s": index / 10, "end_s": (index + 1) / 10,
                     "start_ms": index * 100, "end_ms": (index + 1) * 100})
    (root / f"{stem}_tokens.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8")
    punct = [{"word": "！", "start_s": .3, "end_s": 1.0,
              "start_ms": 300, "end_ms": 1000}]
    if reorder_punct:
        punct = [{"word": "。", "start_s": .3, "end_s": 1.0,
                  "start_ms": 300, "end_ms": 1000}]
    (root / f"{stem}_punct.json").write_text(json.dumps(punct, ensure_ascii=False), encoding="utf-8")
    (root / f"{stem}_ref.txt").write_text(reference.strip() + "\n", encoding="utf-8")
    for suffix in ("_text_raw.txt", "_text_cn.txt"):
        (root / f"{stem}{suffix}").write_text(reference.strip() + "\n", encoding="utf-8")


def test_logits_mask_isolated() -> None:
    logits = torch.full((4, ctc.NVV_END + 1), -10.0)
    logits[:, ctc.BLANK_ID] = 8.0
    logits[:, ctc.NVV_START] = 9.0
    original = logits.clone()
    decoded = ctc._free_decode_logits(logits, reference_only=True, enable_nvv=False,
                                      bias_value=4.0)
    assert torch.equal(logits, original), "free-decode mask mutated clean logits"
    assert decoded[:, ctc.NVV_START].eq(float("-inf")).all()
    assert decoded.argmax(dim=-1).eq(ctc.BLANK_ID).all()
    legacy = ctc._free_decode_logits(logits, reference_only=False, enable_nvv=True,
                                      bias_value=4.0)
    assert legacy.argmax(dim=-1).eq(ctc.NVV_START).all(), "legacy NVV bias was removed"


def test_reference_projection_and_faults() -> None:
    with tempfile.TemporaryDirectory(prefix="reference-only-ctc-") as temp:
        root = Path(temp)
        reference = "你好[Breathing]！"
        write_bundle(root, "demo", reference)
        verifier._actual_projection(root, "demo", reference)
        write_bundle(root, "demo", reference, extra_nvv=True)
        try:
            verifier._actual_projection(root, "demo", reference)
        except ValueError:
            pass
        else:
            raise AssertionError("extra reference NVV was accepted")
        write_bundle(root, "demo", reference, reorder_punct=True)
        try:
            verifier._actual_projection(root, "demo", reference)
        except ValueError:
            pass
        else:
            raise AssertionError("reference punctuation drift was accepted")


def test_command_has_single_no_nvv() -> None:
    args = SimpleNamespace(
        asr_python="python", run_root=Path("/tmp/fresh-reference-only"),
        asr_model="model")
    command = prepare.render_rerun_command(args)
    assert command.count("--no-nvv") == 1
    assert "--overwrite" not in command


def test_wav_axis_pause_gate() -> None:
    with tempfile.TemporaryDirectory(prefix="wav-axis-") as temp:
        out = Path(temp) / "x.TextGrid"
        ctc.write_textgrid([], 1.0, out, pauses=[{
            "start_ms": 400, "end_ms": 1050, "duration_ms": 650,
        }])
        assert "xmax = 1.000000" in out.read_text(encoding="utf-8")
        try:
            ctc.write_textgrid([], 1.0, out, pauses=[{
                "start_ms": 400, "end_ms": 1200, "duration_ms": 800,
            }])
        except ValueError:
            pass
        else:
            raise AssertionError("large pause overrun was silently clipped")


def test_incomplete_all_gpu_shard_uses_isolated_staging() -> None:
    """Read-only planning defers all recovery work until every GPU is safe."""
    with tempfile.TemporaryDirectory(prefix="all-gpu-shard-recovery-") as temp:
        root = Path(temp)
        legacy = root / "_shard_gpu0"
        legacy.mkdir()
        sentinel = legacy / "partial.lab"
        sentinel.write_text("legacy\n", encoding="utf-8")

        staged, reused, recovered = ctc._plan_all_gpu_shard(root, 0, 2)
        assert recovered and not reused
        assert staged.name == "_shard_gpu0_staging"
        assert not staged.exists(), "planning must not create recovery staging"
        assert sentinel.read_text(encoding="utf-8") == "legacy\n"

        # GPU 1 is ambiguous. Planning the full set must fail before GPU 0's
        # planned recovery staging is materialized or any worker is started.
        legacy_1 = root / "_shard_gpu1"
        legacy_1.mkdir()
        (legacy_1 / "partial.lab").write_text("legacy\n", encoding="utf-8")
        (root / "_shard_gpu1_staging").mkdir()
        try:
            ctc._plan_all_gpu_shards(root, [(0, 2), (1, 2)])
        except RuntimeError as exc:
            assert "explicit operator resolution" in str(exc)
        else:
            raise AssertionError("ambiguous later shard was silently accepted")
        assert not staged.exists(), "GPU 0 staging was created before GPU 1 validation"
        assert sentinel.read_text(encoding="utf-8") == "legacy\n"

    with tempfile.TemporaryDirectory(prefix="all-gpu-shard-reuse-") as temp:
        root = Path(temp)
        complete = root / "_shard_gpu0"
        complete.mkdir()
        for stem in ("a", "b"):
            (complete / f"{stem}.lab").write_text("ok\n", encoding="utf-8")
        selected, reused, recovered = ctc._plan_all_gpu_shard(root, 0, 2)
        assert selected == complete and reused and not recovered


def main() -> int:
    test_logits_mask_isolated()
    test_reference_projection_and_faults()
    test_command_has_single_no_nvv()
    test_wav_axis_pause_gate()
    test_incomplete_all_gpu_shard_uses_isolated_staging()
    print("reference-only CTC fault tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
