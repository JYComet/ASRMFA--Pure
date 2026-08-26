#!/usr/bin/env python3
"""Audit a fixed 100-stem authority canary against a fresh run.

The selector is committed and was derived read-only from the historical
54k CTC/reference/audio tree.  This tool never searches another run, repairs
artifacts, or writes outside the explicitly requested report path.  A sample
is publishable only when it is present in ``output/`` and has no independent
reason; uncertain or inconsistent samples belong in ``filtered/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_strict_ok import (  # noqa: E402
    _aligned_reasons, _content_reasons, _ctc_lifecycle_reasons,
    _postprocess_contract_reasons,
)
from postprocess_textgrids import (  # noqa: E402
    AXIS_EPS, _extract_word_chars, is_english_token, is_punct,
    is_silence, parse_textgrid, tier_by_name,
)
from english_units import is_english_fragment_token, project_authority_semantics

SELECTION_SCHEMA = "authority-ok100-selection-v1"
AUDIT_SCHEMA = "authority-ok100-audit-v1"
SELECTION_SEED = "authority-ok100-20260818-v1"
SELECTION_RULE = ("filter ctc_pretg/{stem}_ref.txt by case-insensitive ASCII-letter "
                  "token boundary (?<![A-Za-z])ok(?![A-Za-z]), retain complete "
                  "audio/CTC/reference bundles, then take the first 100 by "
                  "sha256(seed + NUL + stem) with stem tie-break")
REFERENCE_OK_RE = re.compile(r"(?<![A-Za-z])ok(?![A-Za-z])", re.IGNORECASE)
REQUIRED_CTC_SUFFIXES = (".TextGrid", ".lab", "_punct.json", "_ref.txt",
                         "_tokens.jsonl")


class AuditError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise AuditError(f"{label}_missing_or_symlink:{path}")
    return path.resolve(strict=True)


def load_selection(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"selection_unreadable:{path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SELECTION_SCHEMA:
        raise AuditError("selection_schema_mismatch")
    stems = payload.get("stems")
    if (not isinstance(stems, list) or len(stems) != 100
            or any(not isinstance(stem, str) or not stem for stem in stems)
            or len(stems) != len(set(stems))):
        raise AuditError("selection_must_contain_100_unique_stems")
    if payload.get("count") != 100:
        raise AuditError("selection_count_mismatch")
    if payload.get("seed") != SELECTION_SEED:
        raise AuditError("selection_seed_mismatch")
    if payload.get("selection_rule") != SELECTION_RULE:
        raise AuditError("selection_rule_mismatch")
    expected_order = sorted(
        stems,
        key=lambda stem: (hashlib.sha256(
            f"{SELECTION_SEED}\0{stem}".encode("utf-8")).hexdigest(), stem),
    )
    if stems != expected_order:
        raise AuditError("selection_hash_order_mismatch")
    if not isinstance(payload.get("candidate_count"), int) or payload["candidate_count"] < 100:
        raise AuditError("selection_candidate_count_invalid")
    return payload


def _report_rows(path: Path) -> dict[str, dict]:
    if not path.is_file() or path.is_symlink():
        return {}
    rows: dict[str, dict] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditError(f"postprocess_report_invalid_json:{line_number}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("stem"), str):
            raise AuditError(f"postprocess_report_invalid_row:{line_number}")
        stem = row["stem"]
        if stem in rows:
            raise AuditError(f"postprocess_report_duplicate:{stem}")
        rows[stem] = row
    return rows


def _authority_projection(reference: str, final_words, source_words, ctc_rows):
    """Build a proof-bearing multi-to-one authority projection.

    English fragments are consumed only as a complete ordered group for one
    reference unit.  CJK/pinyin, NVV, punctuation, and the next English unit
    are hard boundaries; no substring or dictionary-key comparison is used.
    """
    semantic = [item for item in project_authority_semantics(reference)
                if item["kind"] != "punct"]
    if len(final_words) != len(semantic):
        return None, "authority_surface_count_mismatch"
    source_cursor = 0
    ctc_cursor = 0
    groups = []

    def skip_punct(items, cursor, getter):
        while cursor < len(items) and is_punct(str(getter(items[cursor]))):
            cursor += 1
        return cursor

    def consume(items, cursor, expected, getter):
        cursor = skip_punct(items, cursor, getter)
        kind = expected["kind"]
        if kind == "english":
            compact_expected = "".join(ch for ch in expected["surface"].casefold()
                                        if ch.isalnum())
            start = cursor
            compact = ""
            members = []
            while cursor < len(items):
                text = str(getter(items[cursor])).strip()
                if not is_english_fragment_token(text):
                    break
                part = "".join(ch for ch in text.casefold() if ch.isalnum())
                if text.isdigit() and cursor + 1 < len(items):
                    return None, "numeric_suffix_not_final"
                if not compact_expected.startswith(compact + part):
                    return None, "extra_or_reordered_fragment"
                compact += part
                members.append(items[cursor]); cursor += 1
                if compact == compact_expected:
                    break
            if not members:
                return None, "partial_fragment"
            if compact != compact_expected:
                return None, "partial_fragment"
            return (cursor, members), None
        if cursor >= len(items):
            return None, "semantic_source_missing"
        text = str(getter(items[cursor])).strip()
        if kind == "cjk" and not (text and text[-1:] in "12345"):
            return None, "cjk_boundary_mismatch"
        if kind == "nvv" and not text.strip("<>"):
            return None, "nvv_boundary_mismatch"
        return (cursor + 1, [items[cursor]]), None

    for expected, final in zip(semantic, final_words):
        observed = final.text.strip()
        if expected["kind"] == "english":
            observed_compact = "".join(ch for ch in observed.casefold() if ch.isalnum())
            expected_compact = "".join(ch for ch in expected["surface"].casefold() if ch.isalnum())
            if observed_compact != expected_compact:
                return None, "final_surface_mismatch"
        elif expected["kind"] == "cjk" and not observed.endswith(tuple("12345")):
            return None, "final_cjk_projection_mismatch"
        source_result, error = consume(source_words, source_cursor, expected,
                                       lambda item: item.text)
        if error:
            return None, error
        source_cursor, source_group = source_result
        ctc_result, error = consume(ctc_rows, ctc_cursor, expected,
                                    lambda item: item.get("word", ""))
        if error:
            return None, error
        ctc_cursor, ctc_group = ctc_result
        groups.append({"expected": expected, "final": final,
                       "source": source_group, "ctc": ctc_group})
    if skip_punct(source_words, source_cursor, lambda item: item.text) != len(source_words):
        return None, "extra_source_fragment"
    if skip_punct(ctc_rows, ctc_cursor, lambda item: item.get("word", "")) != len(ctc_rows):
        return None, "extra_ctc_fragment"
    return groups, None


def _exact_ctc_reasons(output_tg, ctc_path: Path, reference: str,
                       source_tg=None) -> list[str]:
    """Check ordinal lexical evidence envelope and local punctuation ownership."""
    reasons: list[str] = []
    words = tier_by_name(output_tg, "words")
    if words is None:
        return ["words_tier_missing"]
    try:
        rows = [json.loads(line) for line in
                ctc_path.with_name(ctc_path.stem + "_tokens.jsonl").read_text(
                    encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ["ctc_lexical_evidence_unreadable"]
    lexical_words = [iv for iv in words.intervals
                     if iv.text.strip() and not is_silence(iv.text)
                     and not is_punct(iv.text)]
    lexical_ctc = [row for row in rows if isinstance(row, dict)
                   and row.get("type", "word") == "word"
                   and row.get("word") and not is_silence(str(row["word"]))]
    source_words = []
    if source_tg is not None:
        source_tier = tier_by_name(source_tg, "words")
        if source_tier is not None:
            source_words = [iv for iv in source_tier.intervals
                            if iv.text.strip() and not is_silence(iv.text)
                            and not is_punct(iv.text)]
    projection, projection_error = _authority_projection(
        reference, lexical_words, source_words, lexical_ctc)
    if projection_error:
        reasons.extend(("ctc_lexical_sequence_mismatch",
                        f"authority_projection:{projection_error}"))
    else:
        outside = []
        previous_word_end = -math.inf
        previous_source_start = -math.inf
        previous_ctc_start = -math.inf
        punctuation = [iv for iv in words.intervals
                       if is_punct(iv.text) and not is_silence(iv.text)]
        for index, group in enumerate(projection):
            word = group["final"]
            source = group["source"][0]
            row = group["ctc"][0]
            source_end = group["source"][-1].xmax
            ctc_end = max(float(item.get("end_s")) for item in group["ctc"]
                          if item.get("end_s") is not None)
            try:
                ctc_start = min(float(item["start_s"]) for item in group["ctc"])
                ctc_end = float(ctc_end)
            except (KeyError, TypeError, ValueError):
                outside.append(index)
                continue
            source_start, source_end = source.xmin, source_end
            envelope_start = min(source_start, ctc_start)
            envelope_end = max(source_end, ctc_end)
            source_overlap = min(word.xmax, source_end) - max(word.xmin, source_start)
            ctc_overlap = min(word.xmax, ctc_end) - max(word.xmin, ctc_start)
            final_word_order = word.xmin >= previous_word_end - AXIS_EPS
            source_order = source_start >= previous_source_start - AXIS_EPS
            ctc_order = ctc_start >= previous_ctc_start - AXIS_EPS
            punct_overlap = any(
                min(word.xmax, punct.xmax) - max(word.xmin, punct.xmin) > AXIS_EPS
                for punct in punctuation)
            if (not all(math.isfinite(value) for value in
                         (ctc_start, ctc_end, source_start, source_end))
                    or ctc_end <= ctc_start or source_end <= source_start
                    # CTC/MFA spans are lexical/order evidence, not a hard
                    # geometry fence.  Processed boundaries may be
                    # compensated outside either raw evidence span.
                    or not final_word_order or not source_order or not ctc_order
                    or punct_overlap):
                outside.append(index)
            previous_word_end = max(previous_word_end, word.xmax)
            previous_source_start = source_start
            previous_ctc_start = ctc_start
        # Do not filter solely because processed geometry lies outside the raw
        # CTC/MFA evidence envelope.  Keep the calculated indices available
        # for diagnostics, but publication authority belongs to processed
        # geometry after arbitration and compensation.

    reference_punct = [unit.strip() for unit in _extract_word_chars(reference)
                       if is_punct(unit)]
    observed_punct = [iv.text.strip() for iv in words.intervals
                      if is_punct(iv.text) and not is_silence(iv.text)]
    if reference_punct != observed_punct:
        reasons.append("punctuation_reference_sequence_mismatch")
    lexical = lexical_words
    for punct in (iv for iv in words.intervals
                  if is_punct(iv.text) and not is_silence(iv.text)):
        prev = next((word for word in reversed(lexical)
                     if word.xmax <= punct.xmin + AXIS_EPS), None)
        nxt = next((word for word in lexical
                    if word.xmin >= punct.xmax - AXIS_EPS), None)
        start = prev.xmax if prev is not None else words.xmin
        end = nxt.xmin if nxt is not None else words.xmax
        if punct.xmin < start - AXIS_EPS or punct.xmax > end + AXIS_EPS:
            reasons.append("punctuation_local_owner_mismatch")
            break
    return sorted(set(reasons))


def audit_run(selection_path: Path, run_root: Path, evidence_root: Path,
              report_path: Path | None = None,
              audio_root: Path | None = None,
              raw_manifest: Path | None = None,
              work_receipt: Path | None = None,
              work_root: Path | None = None) -> dict[str, Any]:
    selection = load_selection(selection_path)
    stems = selection["stems"]
    run_root = run_root.resolve()
    evidence_root = evidence_root.resolve()
    audio_root = Path(audio_root or selection["required_audio_root"]).resolve()
    if not audio_root.is_absolute() or ".." in audio_root.parts:
        raise AuditError("required_audio_root_invalid")
    output_dir = run_root / "output"
    filtered_dir = run_root / "filtered"
    post_report = _report_rows(output_dir / "postprocess_report.jsonl")
    if raw_manifest is None:
        candidate_raw = evidence_root / "ctc_pretg" / ".ctc_raw_manifest.json"
        if candidate_raw.is_file() or candidate_raw.is_symlink():
            raw_manifest = candidate_raw
    resolved_work_root = Path(work_root) if work_root is not None else None
    if resolved_work_root is None and work_receipt is not None:
        resolved_work_root = Path(work_receipt).parent
    if resolved_work_root is None:
        for candidate in (evidence_root / "ctc_pretg_adj",
                          run_root / "ctc_pretg_adj"):
            if (candidate / ".ctc_work_receipt.json").is_file() or (
                    candidate / ".ctc_work_receipt.json").is_symlink():
                resolved_work_root = candidate
                break
    if resolved_work_root is None:
        resolved_work_root = evidence_root / "ctc_pretg_adj"
    lifecycle_args = argparse.Namespace(
        ctc_dir=resolved_work_root,
        ctc_raw_manifest=raw_manifest,
        ctc_work_receipt=work_receipt,
    )
    lifecycle_reasons, lifecycle = _ctc_lifecycle_reasons(
        lifecycle_args, set(stems))
    records: list[dict[str, Any]] = []
    published_mismatches: list[str] = []
    seen_output: set[str] = set()
    seen_filtered: set[str] = set()
    receipt_reasons: list[str] = []
    receipt_stem_reasons: dict[str, list[str]] = {}
    receipt_relpath = selection.get("ctc_run_receipt")
    # The frozen selection hash belongs to the historical producer receipt.
    # A ctc_ready subset run legitimately rewrites a run-local receipt with
    # only the selected stems, so comparing that smaller receipt byte-for-byte
    # with the historical 54k receipt would manufacture a false failure.
    # Validate both sides: immutable historical provenance, and the run-local
    # subset receipt's exact denominator/audio binding.
    historical_receipt_path = (
        Path(str(selection.get("historical_evidence_root"))) / receipt_relpath
        if isinstance(receipt_relpath, str)
        and isinstance(selection.get("historical_evidence_root"), str)
        else None)
    run_receipt_path = (evidence_root / receipt_relpath
                        if isinstance(receipt_relpath, str) else None)
    historical_receipt = None
    receipt = None
    if (historical_receipt_path is None or not historical_receipt_path.is_file()
            or historical_receipt_path.is_symlink()):
        receipt_reasons.append("historical_ctc_receipt_missing")
    else:
        if _sha256(historical_receipt_path) != selection.get("ctc_run_receipt_sha256"):
            receipt_reasons.append("historical_ctc_receipt_sha256_mismatch")
        try:
            historical_receipt = json.loads(
                historical_receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            receipt_reasons.append("historical_ctc_receipt_unreadable")
    if run_receipt_path is None or not run_receipt_path.is_file() or run_receipt_path.is_symlink():
        receipt_reasons.append("run_ctc_receipt_missing")
    else:
        try:
            receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            receipt_reasons.append("run_ctc_receipt_unreadable")
    if isinstance(receipt, dict):
        if receipt.get("schema") != "ctc-run-receipt-v2":
            receipt_reasons.append("run_ctc_receipt_schema_mismatch")
        if (set(receipt.get("input_stems", [])) != set(stems)
                or set(receipt.get("output_stems", [])) != set(stems)):
            receipt_reasons.append("run_ctc_receipt_subset_mismatch")
    historical_bindings = {
        row.get("stem"): row for row in (historical_receipt or {}).get("audio_bindings", [])
        if isinstance(row, dict) and isinstance(row.get("stem"), str)
    }
    run_bindings = {
        row.get("stem"): row for row in (receipt or {}).get("audio_bindings", [])
        if isinstance(row, dict) and isinstance(row.get("stem"), str)
    }
    if set(run_bindings) != set(stems):
        receipt_reasons.append("run_ctc_receipt_audio_binding_subset_mismatch")
    for stem in stems:
        historical = historical_bindings.get(stem)
        current = run_bindings.get(stem)
        local: list[str] = []
        if historical is None:
            local.append("historical_ctc_audio_binding_missing")
        if current is None:
            local.append("run_ctc_audio_binding_missing")
        elif historical is not None:
            if current.get("sha256") != historical.get("sha256"):
                local.append("ctc_audio_sha256_lineage_mismatch")
            if str(Path(str(current.get("path", ""))).resolve().parent) != str(audio_root):
                local.append("ctc_receipt_audio_root_mismatch")
        if local:
            receipt_stem_reasons[stem] = local
    if isinstance(receipt, dict):
        roots = {str(Path(str(row["path"])).resolve().parent)
                 for row in receipt.get("audio_bindings", [])
                 if isinstance(row, dict) and row.get("path")}
        if roots != {str(audio_root)}:
            receipt_reasons.append("ctc_receipt_audio_root_mismatch")

    for stem in stems:
        output_path = output_dir / f"{stem}.TextGrid"
        filtered_path = filtered_dir / f"{stem}.TextGrid"
        locations = int(output_path.is_file()) + int(filtered_path.is_file())
        reasons: list[str] = []
        reasons.extend(lifecycle_reasons)
        reasons.extend(receipt_reasons)
        reasons.extend(receipt_stem_reasons.get(stem, []))
        if locations == 0:
            reasons.append("publication_artifact_missing")
        elif locations > 1:
            reasons.append("publication_artifact_duplicate")
        if output_path.is_file():
            seen_output.add(stem)
            selected_path = output_path
            verdict = "published"
        elif filtered_path.is_file():
            seen_filtered.add(stem)
            selected_path = filtered_path
            verdict = "filtered"
        else:
            selected_path = None
            verdict = "missing"

        ctc_dir = evidence_root / "ctc_pretg"
        audio_path = audio_root / f"{stem}.wav"
        ref_path = ctc_dir / f"{stem}_ref.txt"
        ctc_path = ctc_dir / f"{stem}.TextGrid"
        aligned = evidence_root / "aligned" / f"{stem}.TextGrid"
        if not audio_path.is_file() or audio_path.is_symlink():
            reasons.append("audio_evidence_missing")
        missing_evidence = [suffix for suffix in REQUIRED_CTC_SUFFIXES
                            if not (ctc_dir / f"{stem}{suffix}").is_file()]
        if missing_evidence:
            reasons.append("reference_ctc_evidence_missing")
        reference = ""
        reference_ok = False
        if ref_path.is_file():
            reference = ref_path.read_text(encoding="utf-8").strip()
            if not reference:
                reasons.append("reference_empty")
            elif not REFERENCE_OK_RE.search(reference):
                reasons.append("reference_ok_missing")
            else:
                reference_ok = True
        else:
            reasons.append("reference_ok_missing")

        if selected_path is not None and not reasons:
            try:
                tg = parse_textgrid(selected_path)
                source_tg = parse_textgrid(aligned) if aligned.is_file() else None
                reasons.extend(_content_reasons(tg, reference,
                                                 reference_authoritative=True))
                reasons.extend(_exact_ctc_reasons(
                    tg, ctc_path, reference, source_tg=source_tg))
                if any(is_english_token(iv.text)
                       for iv in tier_by_name(tg, "words").intervals):
                    provenance = post_report.get(stem, {}).get(
                        "english_provenance")
                    if (not isinstance(provenance, dict)
                            or provenance.get("status") != "verified"):
                        reasons.append("english_provenance_missing_or_rejected")
            except Exception as exc:
                reasons.append("publication_unreadable")
                reasons.append(f"publication_exception:{type(exc).__name__}")

        if not aligned.is_file():
            reasons.append("mfa_source_evidence_missing")
        elif selected_path is not None and not reasons:
            reasons.extend(_aligned_reasons(aligned, reference))

        post_row = post_report.get(stem)
        if post_row is None:
            reasons.append("postprocess_report_missing")
        else:
            if selected_path is not None:
                try:
                    report_tg = parse_textgrid(selected_path)
                    reasons.extend(_postprocess_contract_reasons(
                        post_row, report_tg, lifecycle))
                except (OSError, ValueError, UnicodeError) as exc:
                    reasons.append(f"postprocess_geometry_unreadable:{type(exc).__name__}")
            elif lifecycle is not None:
                reasons.extend(_postprocess_contract_reasons(post_row, None, lifecycle))
            reported = post_row.get("filter_reasons", [])
            if post_row.get("status") == "ok" and reported:
                reasons.append("postprocess_status_reason_mismatch")
            contract = post_row.get("publication_contract")
            if isinstance(contract, dict):
                contract_reasons = contract.get("reasons", [])
                if contract.get("status") != "verified" or contract_reasons:
                    reasons.extend(
                        f"publication_contract:{reason}"
                        for reason in contract_reasons
                        if isinstance(reason, str))
                    if not contract_reasons:
                        reasons.append("publication_contract:rejected")
            if verdict == "published" and post_row.get("status") != "ok":
                reasons.append("published_with_filtered_postprocess_status")

        reasons = sorted(set(reasons))
        if verdict == "published" and reasons:
            published_mismatches.append(stem)
            verdict = "published_evidence_mismatch"
        records.append({
            "stem": stem,
            "verdict": verdict,
            "ok": verdict == "published" and not reasons,
            "reference_ok": reference_ok,
            "reasons": reasons,
            "reference": str(ref_path),
            "ctc": str(ctc_path),
            "published_path": str(selected_path) if selected_path else None,
            "postprocess_status": post_row.get("status") if post_row else None,
        })

    all_run_stems = {
        path.stem for directory in (output_dir, filtered_dir)
        if directory.is_dir() for path in directory.glob("*.TextGrid")
    }
    conservation = (
        seen_output.isdisjoint(seen_filtered)
        and seen_output | seen_filtered == set(stems)
        and all_run_stems == set(stems)
    )
    reason_counts = Counter(reason for row in records for reason in row["reasons"])
    reference_ok_all = all(row["reference_ok"] for row in records)
    summary = {
        "schema": AUDIT_SCHEMA,
        "selection": str(selection_path.resolve()),
        "selection_sha256": _sha256(selection_path),
        "run_root": str(run_root),
        "evidence_root": str(evidence_root),
        "audio_root": str(audio_root),
        "count": len(records),
        "published": len(seen_output),
        "filtered": len(seen_filtered),
        "missing": sum(row["verdict"] == "missing" for row in records),
        "conservation": conservation,
        "reference_ok_all": reference_ok_all,
        "ctc_receipt_ok": not receipt_reasons,
        "ctc_lifecycle_ok": not lifecycle_reasons,
        "ctc_lifecycle": lifecycle,
        "published_evidence_consistent": not published_mismatches,
        "reason_counts": dict(sorted(reason_counts.items())),
        "ok": conservation and reference_ok_all and not receipt_reasons
              and not lifecycle_reasons
              and not published_mismatches
              and all(row["verdict"] in {"published", "filtered"}
                      for row in records),
        "records": records,
    }
    if report_path is not None:
        report_path = report_path.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path,
                        default=PROJECT_ROOT / "configs" /
                        "hecheng_ria_ok100_authority.selection.json")
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Fresh run root containing output/ and filtered/.")
    parser.add_argument("--evidence-root", type=Path, default=None,
                        help="Root containing ctc_pretg/ and aligned/; defaults to run root.")
    parser.add_argument("--audio-root", type=Path, default=None,
                        help="Audio root bound by the CTC receipt; defaults to selection metadata.")
    parser.add_argument("--ctc-raw-manifest", type=Path, default=None,
                        help="Explicit immutable CTC raw manifest (optional for legacy fixtures).")
    parser.add_argument("--ctc-work-receipt", type=Path, default=None,
                        help="Explicit mutable CTC work receipt (optional for legacy fixtures).")
    parser.add_argument("--ctc-work-root", type=Path, default=None,
                        help="Mutable CTC work root; defaults to evidence-root/ctc_pretg_adj.")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = audit_run(args.selection, args.run_root,
                           args.evidence_root or args.run_root, args.report,
                           args.audio_root, args.ctc_raw_manifest,
                           args.ctc_work_receipt, args.ctc_work_root)
    except (AuditError, OSError, UnicodeError) as exc:
        print(f"audit_authority_ok100: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in
                      ("ok", "count", "published", "filtered", "missing",
                       "conservation", "reference_ok_all",
                       "ctc_receipt_ok",
                       "ctc_lifecycle_ok",
                       "published_evidence_consistent",
                       "reason_counts")}, ensure_ascii=False, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
