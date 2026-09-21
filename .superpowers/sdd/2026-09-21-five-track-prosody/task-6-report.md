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

## Fix Round 1/5

### Ruling

Expanded the scoped change to `scripts/ja_tts_export.py` only for the runtime
input bridge: the pipeline-declared TTS invocation now accepts only
`ja-prosody-alignment-v1` and has no merge fallback.  Its standalone legacy
fixture path remains until Task 7 migrates the exporter contract.

### Implemented

- Prosody now accepts the normal multi-UID merge aggregate and imports merge
  UID errors into its own partial receipt/ledger.
- Template lookup includes local template ID plus token and alias, and each
  semantic graph is deterministically namespaced before composite assembly.
- Reading and frontend tone evidence are checked per UID/token against the
  semantic locked-reading digest.
- Resume uses an immutable compatibility digest for schema/stage shape while
  mutable configuration continues to drive stage-scoped cache identities.

### Tests

- RED (review reproduction): the previous implementation read only singular
  JSONL, used a template ID alone, and compared the full identity before cache
  scoping; the review’s producer-shaped aggregate, repeated local ID, and
  mutable-resume cases exposed those defects.
- GREEN: `pytest -q tests/test_ja_stage_inputs.py -k 'prosody_handler'` — 2 passed.
- GREEN: `pytest -q tests/test_ja_stage_inputs.py tests/test_ja_resume_identity.py tests/test_ja_tts_export.py` — 18 passed.
- `python -m compileall -q scripts tests` and `git diff --check` — passed.
- `pytest -q` — 1675 passed, 37 skipped, 1 failed in 32.64s; the sole failure remains the identical known Task 8 verifier fixture.

## Fix Round 2/5

- Finding 1: `test_prosody_aggregate_unions_expected_and_blocked_merge_uid_ledger` went RED (incorrect `COMPLETE`), then GREEN; prosody now unions merge expected/blocked/errors and creates a stable propagated blocked error.
- Finding 2/3: composite template lookup is token+alias scoped and graph IDs are namespaced before projection; locked-reading and frontend digest checks remain fail-closed at the token boundary.
- Finding 4: pipeline TTS bridge has no merge fallback and validates prosody schema when `stage_inputs` declares the production handler path; standalone legacy fixture remains explicitly transitional per the Fix Round 1 ruling.
- Finding 5: `test_resume_compatibility_rejects_stage_order_drift` went RED then GREEN; immutable compatibility now contains exact ordered `production_stages` while mutable resources remain cache-scoped.
- Focused: `pytest -q tests/test_ja_stage_inputs.py tests/test_ja_resume_identity.py tests/test_ja_tts_export.py` — 21 passed. `compileall` and `git diff --check` passed.
- Full: `pytest -q` — 1677 passed, 37 skipped, 1 known Task 8 verifier-fixture failure in 32.70s.

## Fix Round 3/5

- A: `test_two_token_producer_local_ids_are_namespaced_with_resolving_refs` checks colliding producer-local mora/basic/template IDs and rewritten phone references.
- B: token-scoped reading/frontend digest checks remain fail-closed; ambiguous occurrence hardening remains a follow-up concern.
- C: `test_tts_stage_consumes_authoritative_alignment_and_declares_real_outputs` exercises a prosody-v1 row through production `stage_inputs`; `test_tts_stage_rejects_merge_v3_on_production_path` rejects merge-v3.
- D: stage cache removes complete `config_digest` before stage-scoped resource calculation; compatibility pins ordered production stages.
- Focused: 23 passed; compile/diff passed. Full: 1679 passed, 37 skipped, 1 identical known Task 8 verifier fixture failure in 32.55s.

## Fix Round 4/5

### RED

- `pytest -q tests/test_ja_stage_inputs.py -k 'two_token_producer or duplicate_occurrences'` exposed stale compatibility/edge references and accepted one UID-only locked-reading row for repeated producer occurrences.
- The prior registered TTS test accepted a schema-labelled row through a direct `native_phones -> phones` shim instead of the `assemble_tts_rows` receipt/partition validator.
- A real `config_identity` artifact mutation test showed `mfa.japanese_dictionary` leaked into semantic-stage identity because filtering looked for the substring `mfa` in a leaf key.

### GREEN

- Graph namespacing now rewrites all authoritative and compatibility identity/reference views: mora/basic/template IDs, `nodes`, `semantic_phone_nodes`, `frontend_phone_nodes`, edges, `target`, and OpenJTalk source/target references. The regression creates two real `openjtalk_to_semantic` graphs with colliding producer-local IDs and checks uniqueness, resolution, and absence of stale local references.
- Occurrence binding now uses the available UID/token/token-alias/alias/candidate/occurrence-index discriminator composite. Reading, frontend, and tone resources must select exactly one occurrence; selected-reading and any persisted locked digest are bound to the semantic digest. Real frontend contracts are revalidated with their producer digest logic before use.
- Added explicit negative regressions for wrong candidate, occurrence index, alias, selected reading, locked-reading digest, frontend digest, and ambiguous UID-only duplicate evidence.
- The registered TTS handler requires a complete prosody-v1 graph, validates its mora/basic/native/duration references, calls `assemble_tts_rows`, then exports the assembled record. The positive test supplies a complete artifact and the negative path tampers `basic_phones`; merge-v3 remains rejected.
- Cache scoping is now an explicit section-to-first-stage dependency map (`asr`, `frontend`, `mfa`, `prosody`) with section-qualified model artifact keys. Real identity mutations prove tone changes begin at prosody, MFA assets at align, and frontend assets at frontend; immutable compatibility still pins schema/stage order for persisted resume validation.

### Verification

- `pytest -q tests/test_ja_stage_inputs.py tests/test_ja_resume_identity.py tests/test_ja_prosody_projection.py tests/test_ja_prosody_schema.py tests/test_ja_tts_export.py tests/test_ja_frontend_contract_w2.py tests/test_ja_phone_adapter_w2.py` — 83 passed, 36 skipped.
- `python -m compileall -q scripts tests` — passed.
- `git diff --check` — passed.
- `pytest -q` — 1687 passed, 37 skipped, 1 failed in 35.62s. The sole failure remains the identical Task 8-owned `tests/test_ja_independent_verifier.py::test_three_uid_bound_fixture_integrity_and_evidence_tamper` fixture mismatch.

### Self-audit

- [x] A: Colliding producer-local IDs are fully remapped across real graph and compatibility references; every checked reference resolves.
- [x] B: UID-only evidence is rejected when repeated occurrences make it ambiguous; selected reading/digest and frontend evidence are occurrence-bound.
- [x] C: The registered handler processes only complete prosody-v1 through the authoritative assembler; malformed/tampered input rejects.
- [x] D: Dependency scoping uses explicit semantic sections, while immutable stage/schema compatibility rejects resume drift.
