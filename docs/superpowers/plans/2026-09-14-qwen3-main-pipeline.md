# Qwen3 Main Pipeline Implementation Plan

**Goal:** Qwen3 provides both main-pipeline transcript modes and normalized timestamps to the existing MFA pipeline.

**Architecture:** Extend the existing six-file producer handoff with one mandatory, versioned timestamp normalizer. Keep acoustic correction and MFA unchanged; carry normalized transcript authority through publication and audit.

**Tech Stack:** Python, native Transformers Qwen3, existing MFA, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-qwen3-main-pipeline-design.md`

## Constraints

Preserve existing dirty work and historical artifacts; no provider fallback; no changes to MFA algorithms or settings; no production corpus rerun. sp0 is less than 0.2 seconds, sp1 less than 0.5, sp2 less than 1.5, otherwise sp3. Use integer microseconds. Long gaps become `…` outside the spoken MFA vocabulary.

## Implementation and verification

- [x] Add normalization regression tests for exact thresholds, leading/trailing silence, idempotence and invalid input. Implement `normalize_timestamps(rows, text, duration)` returning copied lexical rows, text and punctuation.
- [x] Apply it before producer artifact publication. Preserve raw evidence and synchronize English canonical geometry/reference identity. Include implementation/version in identity and cache fingerprints.
- [x] Test reference-only/no-reference routes. Permit reference-only operation without an ASR model tree.
- [x] Migrate main config defaults and known legacy provider options, preserving MFA settings. Add bounded per-GPU Qwen inference workers; preserve item order and per-item failure accounting.
- [x] Reproduce the original-reference overwrite through real `process_one`. Make postprocess and independent audit use verified normalized text; keep original reference evidence.
- [x] Exercise sealed raw artifacts and receipt-bound work copies, including normal postprocessing and publication.
- [x] Update current configuration examples and documentation.
- [x] Run the full suite and final syntax checks, inspect the task-only diff, and report actual hardware verification limits.

Commands:

```bash
python -m pytest -q tests/test_qwen3_timestamp_normalization.py tests/test_qwen3_prealign.py
python -m pytest -q
python -m py_compile scripts/qwen3_timestamp_normalization.py scripts/qwen3_prealign.py scripts/run_pipeline.py scripts/postprocess_textgrids.py scripts/audit_strict_ok.py
```

## Verification result

2026-09-14: `python -m pytest -q` — **1307 passed in 37.21s** (baseline: 1283).
Syntax checks passed. AST comparisons confirm `step_mfa_align`, `step_mfa_align_en`
and default `mfa`, `mfa_en`, `ctc_adjust`, `postprocess` settings are unchanged.
Native runtime preflight passed with Transformers 5.14.1. Local ForcedAligner
model-tree preflight passed in reference-only mode, reporting ASR `not_required`
and no created output. Real GPU inference and real MFA acoustic alignment were
not run; integration tests use a fake Qwen backend and synthetic MFA output,
then exercise sealed raw/work evidence and the real postprocessor.
