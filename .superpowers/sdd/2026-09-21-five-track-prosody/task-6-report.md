# Task 6 implementation report

## Implemented

- Added `assemble_prosody_rows`, which joins merge, semantic, frontend, and
  locked-reading artifacts by UID/token/alias and retains the ordered native
  MFA rows (including duplicate labels and raw interval IDs).
- Added the registered `prosody` handler. It writes canonical JSONL, keeps
  successful UIDs when another UID fails, and records failed UIDs in both the
  receipt and `uid_errors.json` ledger.
- Registered/autoloaded prosody, bridged merge output into prosody, and made
  TTS stage inputs point exclusively to `prosody/prosody_alignments.jsonl`.
- Added `ja_prosody.py` to implementation identity and scoped stage cache
  identity so prosody-resource changes start at prosody while MFA dictionary
  changes start at align.
- Added a non-destructive `--workspace` override. Existing non-resume targets
  fail normal resume validation before any content is replaced.

## TDD evidence

### RED

1. `pytest -q tests/test_ja_stage_inputs.py -k 'assemble_prosody or prosody_handler'`
   failed collection with `ImportError: cannot import name 'assemble_prosody_rows'`.
2. `pytest -q tests/test_ja_resume_identity.py -k 'prosody_resources or mfa_dictionary or workspace_override or autoload_registers'`
   failed four expected behaviors: upstream cache identities changed,
   `--workspace` was unrecognized, and prosody remained a skeleton.
3. `pytest -q tests/test_ja_stage_inputs.py -k 'tts_assembly_rejects_pre_prosody'`
   failed because a `ja-en-alignment-v3` row was accepted by TTS assembly.
4. `pytest -q tests/test_ja_resume_identity.py -k 'tts_stage_input_is_only'`
   failed with `KeyError: 'tts'`, proving the bridge did not populate the
   authoritative prosody artifact.

### GREEN

- `pytest -q tests/test_ja_stage_inputs.py tests/test_ja_resume_identity.py` — 15 passed.
- `pytest -q tests/test_ja_stage_inputs.py tests/test_ja_resume_identity.py tests/test_ja_prosody_projection.py tests/test_ja_tts_export.py` — 49 passed.
- `python -m compileall -q scripts tests` — passed.
- `git diff --check` — passed.

## Final verification

- `pytest -q` — **1674 passed, 37 skipped, 1 failed in 35.63s**.
  The only failure is the known Task 8-owned
  `tests/test_ja_independent_verifier.py::test_three_uid_bound_fixture_integrity_and_evidence_tamper`.
  It remains a pre-existing five-track verifier fixture mismatch and is not
  modified by this task.

## Self-review

- Composite joins do not use phone labels as keys, avoiding duplicate-label
  corruption.
- The handler catches errors per UID and never uses an alignment-stage fallback
  for TTS.
- The known Task 8 verifier fixture failure is expected to remain if it is the
  sole full-suite failure; final output below records the observed result.
