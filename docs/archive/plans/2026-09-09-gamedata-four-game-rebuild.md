# GAMEDATA Four-Game Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild only reverse1999 and the three newly discovered games from the current GAMEDATA source, then publish validated speaker-structured TextGrids and silence-normalized WAVs without changing the other seven games.

**Architecture:** A task-level YAML drives recursive inventory, deterministic staging, mixed reference/ASR policy, per-game pipeline configs, and journaled publication. Accepted pipeline output is built in an isolated NVMe run and NAS publish staging; a read-only verifier approves both trees before the four target directories are archived and atomically replaced.

**Tech Stack:** Python 3, PyYAML, ffmpeg, soundfile, existing CTC/MFA pipeline, pytest, POSIX `os.replace` journals.

**Spec:** `docs/superpowers/specs/2026-09-09-gamedata-speaker-finalization-design.md` plus the approved Terra execution map for the four-game rebuild.

## Global Constraints

- Process only `reverse1999`, `persona5x`, `pgr`, and `wuhua`; preserve the seven existing game trees byte-for-byte.
- Never modify `/mnt/Raw/GAMEDATA` source data.
- Reverse1999 must use a fresh source manifest from `/mnt/Raw/GAMEDATA/重返未来1999`; do not reuse prior accepted TextGrids.
- Reference policy is per stem: nonempty same-directory same-basename `.txt` wins; missing/empty text uses NVASR.
- Decode every supported source format to mono PCM16 WAV and normalize edge silence to 0.5 seconds.
- Build and verify in staging before replacing public directories; retain timestamped archives and rollback journal.
- Preserve unrelated dirty worktree changes; use `apply_patch` for repository edits.

---

### Task 1: Configuration and inventory engine

**Files:**
- Create: `configs/gamedata_rebuild_20260909.yaml`
- Create: `scripts/rebuild_gamedata.py`
- Test: `tests/test_rebuild_gamedata.py`

**Interfaces:**
- `load_task_config(path: Path) -> dict`
- `validate_task_config(config: dict) -> None`
- `inventory_game(source_dir: Path, config: dict) -> InventoryReport`
- `resolve_stem(relative_path: Path, collisions: Mapping[str, Sequence[Path]]) -> str`
- `stage_audio(item: InventoryItem, destination: Path, ffmpeg: Path) -> StageResult`
- `write_resolved_pipeline_config(config: dict, game: GameConfig, run_root: Path) -> Path`

- [ ] **Step 1: Add failing tests** for absolute-root validation, target/preserved game allowlists, same-directory nonempty reference selection, empty/missing text fallback, unsupported-extension reporting, case-folded collision naming, and source symlink rejection.
- [ ] **Step 2: Run the focused tests** with `PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_rebuild_gamedata.py -q` and confirm the new APIs fail.
- [ ] **Step 3: Add the task YAML** recording source/stage/work/output/archive roots, four source-to-output mappings, supported extensions, transcript policy, speaker rule, PCM16 mono conversion, 0.5-second edge silence, and the seven preserved codenames.
- [ ] **Step 4: Implement inventory and staging** with deterministic sorted traversal, exact same-directory text pairing, SHA-256 collision suffixes, temporary ffmpeg output plus `soundfile.info` validation, atomic rename, and per-item provenance/report JSON.
- [ ] **Step 5: Implement resolved pipeline config generation** using `reference_mode: auto`, `ctc_prealign.allow_missing_reference: true`, fresh run-specific data/workspace/output paths, and `mfa_en.strict_provenance: false` for fallback compatibility.
- [ ] **Step 6: Rerun focused tests and `git diff --check`**; expected result is all inventory/staging tests green and no whitespace errors.

### Task 2: Dynamic speaker finalization and journaled publication

**Files:**
- Modify: `scripts/finalize_gamedata_speakers.py`
- Test: `tests/test_finalize_gamedata_speakers.py`

**Interfaces:**
- `load_game_specs_from_task(config: dict, run_receipts: Mapping[str, Path]) -> list[GameSpec]`
- `build_publish_staging(spec: GameSpec, staging_root: Path) -> PublishReceipt`
- `archive_targets(specs: Sequence[GameSpec], archive_root: Path, journal: Path) -> None`
- `publish_targets(specs: Sequence[GameSpec], staging_root: Path, journal: Path) -> None`
- `rollback_targets(journal: Path) -> None`

- [ ] **Step 1: Add failing tests** for dynamic specs, exact four-game path allowlisting, non-target immutability, symlink rejection, dual-tree staging, archive journal creation, partial publish rollback, and legacy `production_specs()` compatibility.
- [ ] **Step 2: Run the focused finalizer tests** and confirm the dynamic/staging cases fail while documenting any existing behavior that must remain green.
- [ ] **Step 3: Implement config-driven spec loading** while leaving the current eight-game CLI behavior intact when no task config is supplied.
- [ ] **Step 4: Implement staging publication** that maps accepted manifest entries to `<game>/<speaker>/<stem>`, reuses or generates padded WAVs, shifts generated TextGrids by the actual head offset, and writes per-game receipt counts.
- [ ] **Step 5: Implement archive/publish/rollback journal** with exact-root checks and `os.replace`; refuse root directories, symlinks, paths outside configured roots, or any non-target game.
- [ ] **Step 6: Run both finalizer and rebuild focused tests plus `git diff --check`**; expected result is green tests and a clean diff check.

### Task 3: Preflight inventory, MFA capability probe, and smoke runs

**Files:**
- Runtime only: `/mnt/nvme3/gamedata_rebuild_20260909/<run_id>/`
- Runtime only: `/mnt/Raw/.gamedata_publish_staging/<run_id>/`
- Runtime only: `/mnt/Raw/.gamedata_rebuild_archive/<run_id>/`

- [ ] **Step 1: Record immutable baselines** for the seven preserved game trees in both output roots, including counts, relative-path digest, and classification receipt hashes.
- [ ] **Step 2: Run `rebuild_gamedata.py --validate-config`** and stop if any configured path is outside its allowlisted root or any target resolves through a symlink.
- [ ] **Step 3: Run `rebuild_gamedata.py --inventory-only`** and record discovered, reference, ASR, conversion-error, orphan-text, and collision counts for all four targets.
- [ ] **Step 4: Probe the selected MFA runtime** for the anchor/output interface used by the pipeline; stop before data processing if the runtime rejects required arguments.
- [ ] **Step 5: Run a bounded smoke item** for each available transcript policy in each game, verify TextGrid parsing and WAV readability, and stop on any smoke failure.

### Task 4: Fresh four-game production into staging

**Files:**
- Runtime only under the run root and publish staging roots from Task 3.

- [ ] **Step 1: Create a unique run ID** and stage all four source trees with fresh manifests; ensure reverse1999 manifest paths point only to the current source directory.
- [ ] **Step 2: Convert all staged audio** to validated mono PCM16 WAV and record conversion provenance, source speaker, reference/ASR policy, and source checksum.
- [ ] **Step 3: Run each resolved pipeline config** with all GPUs confirmed free at launch, writing accepted/filtered/error reports to the isolated run root and never to public output directories.
- [ ] **Step 4: Build speaker-structured TextGrid and GAMESL staging trees** from accepted results only; normalize edge silence to 0.5 seconds and keep TextGrid offsets synchronized.
- [ ] **Step 5: Write execution receipts** with source accounting, accepted/filtered/error counts, per-game duration seconds/hours, and immutable-game fingerprints; do not archive or publish yet.

### Task 5: Read-only staging verification

**Files:**
- Create/update: `handoffs/20260909-gamedata-four-game-rebuild.md`

- [ ] **Step 1: Parse every staged TextGrid** and require valid bounds and the established five tiers.
- [ ] **Step 2: Open every staged WAV** and require positive frames/sample rate, mono PCM16 format, and the configured edge-silence tolerance.
- [ ] **Step 3: Compare relative path sets** between staged TextGrids and WAVs and reconcile them with per-item manifests and receipts.
- [ ] **Step 4: Recompute preserved-game fingerprints** and reject any difference from Task 3 baselines.
- [ ] **Step 5: Record `STAGING_APPROVED`** only when all checks pass; otherwise record exact failures and stop before publication.

### Task 6: Journaled archive, publish, and final acceptance

**Files:**
- Runtime only: `/mnt/Raw/.gamedata_rebuild_archive/<run_id>/`
- Runtime only: `/mnt/Raw/.gamedata_publish_staging/<run_id>/`
- Runtime only: four target public game directories

- [ ] **Step 1: Recheck preserved-game fingerprints** and staging receipts immediately before replacement.
- [ ] **Step 2: Archive exactly the four existing target game directories** under the run-specific archive using `os.replace`; never delete them.
- [ ] **Step 3: Publish the verified aligned and GAMESL trees** with journaled `os.replace`; on partial failure execute rollback and stop.
- [ ] **Step 4: Run final read-only verification** against published trees, archives, receipts, and journal; confirm preserved games remain unchanged.
- [ ] **Step 5: Append final accepted counts, filtered/error counts, and total hours** to the handoff; confirm no rebuild, ffmpeg, run_pipeline, or finalizer processes remain.
- [ ] **Step 6: Run focused tests and `git diff --check`** one final time before reporting completion.
