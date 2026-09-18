# GAMEDATA Speaker Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the accepted outputs for all eight games by original speaker and create matching padded WAVs under GAMESL.

**Architecture:** A dataset-specific finalizer reads the frozen stage manifests, plans every source-to-destination mapping, validates the complete plan, then performs idempotent moves/copies and writes an atomic per-game receipt. Existing padded WAVs are reused; missing padded WAVs are generated with the repository's established edge-silence algorithm.

**Tech Stack:** Python 3.11+, pathlib, concurrent.futures, soundfile, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-gamedata-speaker-finalization-design.md`

## Global Constraints

- Do not rerun reverse1999 alignment for four exceptional rows (0.01009%).
- Only report-classified `ok`/published TextGrids enter accepted output.
- Speaker comes from the original audio parent; unresolved stems use `default`.
- TextGrid and GAMESL WAV stem sets must match exactly.
- Preserve root metadata and all pre-existing recovery evidence.

---

### Task 1: Classification planner and padding primitive

**Files:**
- Create: `scripts/finalize_gamedata_speakers.py`
- Create: `tests/test_finalize_gamedata_speakers.py`

**Interfaces:**
- Consumes: stage manifest JSON and accepted TextGrid roots.
- Produces: `speaker_for_stem`, `normalize_edge_silence`, and a fail-closed per-game plan.

- [ ] Write failing tests for original-parent speaker mapping, `default` fallback,
      collision rejection, and 0.5-second edge normalization.
- [ ] Run the focused tests and confirm they fail because the module is absent.
- [ ] Implement the minimal pure helpers and planning validation.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Idempotent execution and receipts

**Files:**
- Modify: `scripts/finalize_gamedata_speakers.py`
- Modify: `tests/test_finalize_gamedata_speakers.py`

**Interfaces:**
- Consumes: the validated plan from Task 1.
- Produces: classified TextGrids, GAMESL WAVs, and `.speaker_classification_receipt.json`.

- [ ] Write failing tests for move/copy execution, generated-audio TextGrid shift,
      rerun idempotence, and conflicting destination rejection.
- [ ] Run the focused tests and confirm the expected failures.
- [ ] Implement bounded parallel execution and atomic receipt writing.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Dataset execution and end-to-end verification

**Files:**
- Modify: `docs/mfa_pipeline_audit_20260907.md`

**Interfaces:**
- Consumes: the eight configured source sets in the finalizer.
- Produces: the final NAS layout and audit record.

- [ ] Run `--dry-run` and verify an aggregate plan of exactly 151,877 accepted stems.
- [ ] Execute the finalizer and retain its per-game receipts.
- [ ] Verify per-game TextGrid/WAV set equality, parse all TextGrids, read all WAV
      headers, and confirm no flat accepted TextGrids remain.
- [ ] Record final counts and exception policy in the audit document.
- [ ] Run focused tests, relevant regression tests, and `git diff --check`.

