# Japanese/English Five-Track Prosody Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce versioned Japanese/English training JSON and an exactly five-tier TextGrid in which native MFA intervals are authoritatively linked to kana, semantic basic phones, and auditable mora-level H/L/UNK tone.

**Architecture:** Keep MFA's merged native-phone timeline immutable, enrich the existing semantic graph with one-mora basic-phone nodes, and add a pure `prosody` stage between `merge` and `tts`. The prosody stage resolves contextual mora tone from versioned evidence, projects ordered mora labels onto native MFA intervals without inventing boundaries, and hands a self-contained v1 prosody alignment to a v2 TTS exporter and an independent verifier.

**Tech Stack:** Python 3.10+, standard library (`dataclasses`, `hashlib`, `json`, `pathlib`, `re`, `wave`), PyYAML, pytest, existing pinned `pyopenjtalk-plus` frontend runtime, Montreal Forced Aligner artifacts, Praat long-text TextGrid.

**Spec:** `docs/superpowers/specs/2026-09-21-ja-en-five-track-prosody-design.md`

## Global Constraints

- New production artifacts use exactly `ja-semantic-phone-graph-v2`, `ja-en-alignment-v3`, `ja-prosody-alignment-v1`, `tts-training-record-v2`, and `five-track-textgrid-v1`.
- The production order is exactly `inventory → audio → asr → reading → frontend → semantic → anchors → align → merge → prosody → tts → verify`; Julius remains diagnostic-only.
- MFA is the only production source of native phone boundaries; prosody must not alter a phone's integer sample boundaries, alias, dictionary pronunciation, raw interval ID, or run provenance.
- Track names and order are exactly `original_text`, `kana`, `mfa_phone`, `phone_kana`, `phone_tone`.
- `mfa_phone`, `phone_kana`, and `phone_tone` have identical authoritative interval count and integer sample boundaries.
- Each Japanese basic phone belongs to exactly one mora; one native phone may cover several ordered basic phones and moras, and such a group has no inferred internal boundary.
- Manual override outranks fixed accent lexicon, which outranks contextual frontend prediction, which outranks `UNK`; every known H/L value has reopenable provenance.
- English, silence, and non-language events use `NA` tone and no Japanese mora; unknown Japanese tone remains `UNK`.
- Text tone and measured F0 remain separate; no-F0 never implies `L`, and nearest-neighbor H/L filling is forbidden.
- A locked reading change invalidates contextual frontend accent evidence unless an override or lexicon entry is explicitly bound to the new locked-reading digest.
- New runtime dependencies are not added; the pinned frontend provider and MFA model/dictionary identities remain part of cache and receipt identity.
- The legacy Chinese pipeline, its normalization, phone maps, stage order, and publish entry points are not modified.

## Review Focus

- Leading/trailing whitespace and punctuation-only spans must reconstruct the original source text while all five TextGrid tiers continuously cover `[0, xmax]` with legal empty intervals; Task 7 pins this.
- Repeated identical aliases and phone labels must join by UID, occurrence, raw interval ID, and pronunciation order rather than set membership or label lookup; Tasks 4 and 6 pin this.
- Pure-English and Japanese-free event records must contain no mora/basic-phone claims and must project `phone_kana=""`, `phone_tone="NA"`; Tasks 5 and 7 pin this.
- A development run may contain only `UNK` Japanese tones, while `publish.require_all_tones=true` must reject it without modifying or rerunning MFA outputs; Tasks 5 and 8 pin this.
- Quoted Unicode labels and adjacent one-sample intervals must survive TextGrid escaping/parsing and preserve exact integer sample boundaries; Task 7 pins this.

---

## File and Interface Map

- `scripts/ja_en_schema.py` owns version names, required fields, stable error codes, stage ordering, and config validation.
- `scripts/ja_frontend.py` owns extraction and persistence of contextual full-context accent evidence; it never chooses a replacement for a locked reading.
- `scripts/ja_phone_adapter.py` owns the semantic v2 graph: mora nodes, one-mora basic-phone nodes, native-phone templates, and explicit phonological transforms.
- `scripts/merge_ja_en_mfa.py` owns the immutable `ja-en-alignment-v3` native timeline and raw-MFA provenance.
- `scripts/ja_prosody.py` is new and owns source-priority resolution, phrase-to-mora tone conversion, native-phone projection, and the prosody stage receipt.
- `scripts/ja_en_stage_inputs.py` owns cross-stage joins by stable IDs and prepares `prosody` and `tts` inputs.
- `scripts/ja_tts_export.py` owns `tts-training-record-v2` and the five-tier TextGrid writer, but does not derive tone.
- `scripts/verify_ja_en_tts.py` reopens external artifacts and independently recomputes relationships and hashes.
- `scripts/run_ja_en_pipeline.py` owns stage registration, autoloading, input bridging, and cache identities.

## Task 1: Version the contracts and configuration surface

**Files:**
- Modify: `scripts/ja_en_schema.py:12-145,385-520`
- Modify: `configs/japanese_english_tts.yaml`
- Modify: `tests/test_ja_en_foundation.py`
- Modify: `tests/test_ja_resume_identity.py`

**Interfaces:**
- Consumes: existing `validate_config(config: Mapping[str, Any]) -> dict[str, Any]`, `PRODUCTION_STAGES`, `SCHEMAS`, `ERROR_CODES`.
- Produces: validated `config["prosody"]`, the five new schema identifiers, six new stable errors, and `prosody` in the production stage tuple before `tts`.

- [ ] **Step 1: Write contract tests that name every new public identifier**

```python
def test_five_track_contract_versions_and_stage_order():
    assert {
        "ja-semantic-phone-graph-v2", "ja-en-alignment-v3",
        "ja-prosody-alignment-v1", "tts-training-record-v2",
        "five-track-textgrid-v1",
    } <= SCHEMAS
    assert PRODUCTION_STAGES[-4:] == ("merge", "prosody", "tts", "verify")
    assert {
        "accent_phrase_unresolved", "tone_cardinality_mismatch",
        "native_basic_mapping_ambiguous", "phone_tone_projection_lossy",
        "five_track_boundary_mismatch", "tone_provenance_missing",
    } <= ERROR_CODES

def test_prosody_config_has_closed_keys(tmp_path):
    config = minimal_config(tmp_path)
    config["prosody"] = {
        "algorithm_version": "ja-mora-tone-v1",
        "manual_overrides": None,
        "accent_lexicon": None,
        "allow_unknown_tones": True,
    }
    assert validate_config(config)["prosody"]["algorithm_version"] == "ja-mora-tone-v1"
    config["prosody"]["nearest_neighbor_fill"] = True
    with pytest.raises(JAContractError, match="config_unknown_key"):
        validate_config(config)
```

- [ ] **Step 2: Run the focused tests and confirm the old contract fails**

Run: `pytest -q tests/test_ja_en_foundation.py tests/test_ja_resume_identity.py`

Expected: FAIL because `prosody` and the new schemas/error codes are absent.

- [ ] **Step 3: Add exact schema requirements and a closed prosody config**

```python
PRODUCTION_STAGES = (
    "inventory", "audio", "asr", "reading", "frontend", "semantic",
    "anchors", "align", "merge", "prosody", "tts", "verify",
)

SCHEMA_REQUIRED_FIELDS.update({
    "ja-semantic-phone-graph-v2": frozenset(
        {"schema", "uid", "mora_nodes", "basic_phone_nodes", "native_phone_templates", "edges"}
    ),
    "ja-en-alignment-v3": frozenset(
        {"schema", "uid", "words", "native_phones", "languages", "raw_mfa"}
    ),
    "ja-prosody-alignment-v1": frozenset(
        {"schema", "uid", "words", "moras", "basic_phones", "native_phones", "tone_sources"}
    ),
    "tts-training-record-v2": frozenset(
        {"schema", "uid", "train_wav", "alignment_wav", "words", "native_phones", "moras", "basic_phones", "quality_masks"}
    ),
    "five-track-textgrid-v1": frozenset(
        {"schema", "uid", "sample_rate", "frame_count", "tiers", "source_json_sha256"}
    ),
})

PROSODY_KEYS = frozenset({
    "algorithm_version", "manual_overrides", "accent_lexicon",
    "allow_unknown_tones",
})
```

Validate resource paths with the existing absolute-path and file-hash helpers. Set the committed development config to `algorithm_version: ja-mora-tone-v1`, null resources, and `allow_unknown_tones: true`; put the release gate under existing `publish.require_all_tones: false`.

- [ ] **Step 4: Pin resume drift to stage sequence and schema versions**

```python
def test_legacy_workspace_is_stale_after_prosody_stage_is_added(tmp_path):
    config_path, config = _config(tmp_path)
    workspace = Path(config["workspace"])
    workspace.mkdir(parents=True)
    legacy_identity = {"schema": "ja-en-run-identity-v1", "production_stages": [
        "inventory", "audio", "asr", "reading", "frontend", "semantic",
        "anchors", "align", "merge", "tts", "verify",
    ]}
    atomic_write_json(workspace / ".ja_en_run_identity.json", {
        **legacy_identity, "identity_digest": stable_digest(legacy_identity)
    }, workspace=workspace)
    _, manifest_path, rows, _ = preflight(config, config_path=config_path)
    with pytest.raises(JAContractError) as error:
        validate_resume(workspace, config_identity(config, manifest_path, rows), allow_new=False)
    assert error.value.code in {"resume_identity_drift", "resume_stale"}
```

Run: `pytest -q tests/test_ja_en_foundation.py tests/test_ja_resume_identity.py`

Expected: PASS.

- [ ] **Step 5: Commit the contract boundary**

```bash
git add scripts/ja_en_schema.py configs/japanese_english_tts.yaml tests/test_ja_en_foundation.py tests/test_ja_resume_identity.py
git commit -m "feat: version five-track prosody contracts"
```

## Task 2: Persist contextual accent evidence from the pinned frontend

**Files:**
- Modify: `scripts/ja_frontend.py:40-210,500-760`
- Modify: `tests/test_ja_frontend_contract_w2.py`
- Modify: `tests/test_ja_reading_selector.py`

**Interfaces:**
- Consumes: pinned worker result from `run_frontend_detailed(text)` and `extract_fullcontext(text)`, plus the existing locked-reading digest.
- Produces: `extract_contextual_accent_evidence(njd_rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, Any]`; each frontend unit contains `accent_evidence`, `accent_evidence_valid`, and `locked_reading_digest`.

- [ ] **Step 1: Add fixtures for flat, head-high, middle-high, tail-high, and multiword phrases**

```python
@pytest.mark.parametrize(("phrase", "nucleus", "expected"), [
    ({"mora_count": 3, "nucleus": 0}, 0, ["L", "H", "H"]),
    ({"mora_count": 3, "nucleus": 1}, 1, ["H", "L", "L"]),
    ({"mora_count": 4, "nucleus": 2}, 2, ["L", "H", "L", "L"]),
    ({"mora_count": 4, "nucleus": 3}, 3, ["L", "H", "H", "L"]),
])
def test_phrase_tone_fixture_shape(phrase, nucleus, expected):
    evidence = accent_fixture(phrase["mora_count"], nucleus)
    assert evidence["expected_tones"] == expected

def test_multiword_phrase_keeps_one_phrase_id(frontend_fixture):
    result = extract_contextual_accent_evidence(
        frontend_fixture["njd_rows"], frontend_fixture["full_context_labels"]
    )
    assert [row["accent_phrase_id"] for row in result["moras"]] == ["ap0", "ap0", "ap0", "ap0"]
    assert result["provider_evidence_sha256"] == stable_digest({
        "njd_rows": frontend_fixture["njd_rows"],
        "full_context_labels": frontend_fixture["full_context_labels"],
    })
```

The fixture labels are committed literal outputs from the already pinned frontend runtime, not generated during tests.

- [ ] **Step 2: Run the frontend contract tests and observe the missing API**

Run: `pytest -q tests/test_ja_frontend_contract_w2.py tests/test_ja_reading_selector.py`

Expected: FAIL importing `extract_contextual_accent_evidence`.

- [ ] **Step 3: Extend the provider worker and normalize evidence**

```python
_CAPABILITIES = (
    "g2p_mapping", "run_frontend_detailed", "make_phoneme_mapping",
    "extract_fullcontext",
)

def extract_contextual_accent_evidence(njd_rows, labels):
    rows = [dict(row) for row in njd_rows]
    groups, current = [], []
    for row in rows:
        if current and int(row.get("chain_flag", 0)) != 1:
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)
    moras = _bind_full_context_moras(groups, list(labels))
    return {
        "adapter_version": "openjtalk-fullcontext-accent-v1",
        "accent_phrases": _phrase_summaries(moras),
        "moras": moras,
        "full_context_labels": list(labels),
        "provider_evidence_sha256": stable_digest({
            "njd_rows": rows, "full_context_labels": list(labels)
        }),
    }
```

`_bind_full_context_moras` parses OpenJTalk label features into explicit mora sequence, phrase boundary, and nucleus/downstep fields, validates monotonic positions and cardinality, and raises `JAContractError("accent_phrase_unresolved", ...)` rather than guessing if labels and NJD disagree.

- [ ] **Step 4: Invalidate contextual accent when reading is overridden**

```python
if unit["locked_reading_digest"] != unit["contextual_reading_digest"]:
    unit["accent_evidence_valid"] = False
    unit["accent_evidence_invalid_reason"] = "locked_reading_changed"
else:
    unit["accent_evidence_valid"] = True
```

Add this exact regression:

```python
def test_reading_override_invalidates_contextual_accent(contract):
    contract["units"][0]["locked_reading"] = "トウキョウ"
    contract["units"][0]["locked_reading_digest"] = stable_digest("トウキョウ")
    checked = validate_frontend_contract(contract)
    assert checked["units"][0]["accent_evidence_valid"] is False
    assert checked["units"][0]["accent_evidence_invalid_reason"] == "locked_reading_changed"
```

- [ ] **Step 5: Verify deterministic evidence with the pinned runtime and commit**

Run: `pytest -q tests/test_ja_frontend_contract_w2.py tests/test_ja_reading_selector.py`

Expected: PASS, including a subprocess test that runs the pinned frontend twice and compares the evidence digest.

```bash
git add scripts/ja_frontend.py tests/test_ja_frontend_contract_w2.py tests/test_ja_reading_selector.py
git commit -m "feat: retain contextual Japanese accent evidence"
```

## Task 3: Build the semantic v2 two-layer phone graph

**Files:**
- Modify: `scripts/ja_phone_adapter.py:1-430`
- Modify: `tests/test_ja_phone_adapter_w2.py`
- Modify: `tests/test_ja_mora_graph.py`

**Interfaces:**
- Consumes: `openjtalk_to_semantic(unit: Mapping[str, Any]) -> dict[str, Any]` inputs with locked kana and frontend phone mapping.
- Produces: the same function returning a `ja-semantic-phone-graph-v2` object with `mora_nodes`, `basic_phone_nodes`, `native_phone_templates`, and `edges`; every native template has ordered `basic_phone_ids`, ordered `mora_ids`, and one allowed `transform`.

- [ ] **Step 1: Express the special-phonology table as graph tests**

```python
@pytest.mark.parametrize(("reading", "native", "symbols", "moras", "transform"), [
    ("コー", "oː", ["o", "o"], ["コ", "ー"], "long_vowel_merge"),
    ("キット", "tː", ["Q", "t"], ["ッ", "ト"], "geminate_merge"),
    ("オンナ", "nː", ["N", "n"], ["ン", "ナ"], "nasal_coalescence"),
    ("グッズ", "dzː", ["Q", "dz"], ["ッ", "ズ"], "geminate_merge"),
])
def test_native_template_preserves_order(reading, native, symbols, moras, transform):
    graph = openjtalk_to_semantic(frontend_unit(reading))
    template = next(row for row in graph["native_phone_templates"] if row["native_phone"] == native)
    basic = {row["basic_phone_id"]: row for row in graph["basic_phone_nodes"]}
    mora = {row["mora_id"]: row for row in graph["mora_nodes"]}
    assert [basic[key]["symbol"] for key in template["basic_phone_ids"]] == symbols
    assert [mora[key]["kana"] for key in template["mora_ids"]] == moras
    assert template["transform"] == transform

def test_each_basic_phone_has_exactly_one_mora():
    graph = openjtalk_to_semantic(frontend_unit("ウンメー"))
    assert graph["basic_phone_nodes"]
    assert all(isinstance(row["mora_id"], str) for row in graph["basic_phone_nodes"])
    assert len({row["basic_phone_id"] for row in graph["basic_phone_nodes"]}) == len(graph["basic_phone_nodes"])
```

- [ ] **Step 2: Run the semantic tests and confirm v1 shape fails**

Run: `pytest -q tests/test_ja_phone_adapter_w2.py tests/test_ja_mora_graph.py`

Expected: FAIL because the current graph has generic nodes/edges and no basic-phone layer.

- [ ] **Step 3: Add typed constructors and closed roles/transforms**

```python
MORA_KINDS = frozenset({
    "regular", "long_extension", "sokuon", "nasal_mora",
    "final_sokuon", "devoiced", "elided",
})
BASIC_ROLES = frozenset({
    "onset", "nucleus", "long_extension", "sokuon",
    "nasal_mora", "final_sokuon",
})
TRANSFORMS = frozenset({
    "identity", "long_vowel_merge", "geminate_merge",
    "nasal_coalescence", "devoiced_realization", "final_sokuon",
})

def _basic(node_id, symbol, role, mora_id, realization="observed"):
    if role not in BASIC_ROLES or realization not in {
        "observed", "merged", "devoiced", "elided", "unresolved"
    }:
        raise JAContractError("schema_invalid", "invalid basic phone role or realization")
    return {
        "basic_phone_id": node_id, "symbol": symbol, "role": role,
        "mora_id": mora_id, "realization": realization,
        "native_phone_id": None,
    }
```

Use existing `_target_phones` and `_source_mora_indices` only to select model-native symbols; build explicit ordered nodes and reject ambiguous relations with `native_basic_mapping_ambiguous`.

- [ ] **Step 4: Preserve devoiced, elided, and final-sokuon nodes without fake time**

```python
def test_elided_vowel_has_no_native_interval():
    graph = openjtalk_to_semantic(frontend_unit("スキ", elide="u"))
    vowel = next(row for row in graph["basic_phone_nodes"] if row["symbol"] == "u")
    assert vowel["realization"] == "elided"
    assert vowel["native_phone_id"] is None

def test_final_sokuon_exists_only_when_locked_reading_contains_it():
    graph = openjtalk_to_semantic(frontend_unit("アッ"))
    assert graph["mora_nodes"][-1]["kind"] == "final_sokuon"
    assert graph["basic_phone_nodes"][-1]["role"] == "final_sokuon"
```

Run: `pytest -q tests/test_ja_phone_adapter_w2.py tests/test_ja_mora_graph.py tests/test_ja_phone_adapter.py`

Expected: PASS.

- [ ] **Step 5: Commit the graph migration**

```bash
git add scripts/ja_phone_adapter.py tests/test_ja_phone_adapter_w2.py tests/test_ja_mora_graph.py tests/test_ja_phone_adapter.py
git commit -m "feat: model basic and native Japanese phones"
```

## Task 4: Emit alignment v3 and bind native phones without positional fallbacks

**Files:**
- Modify: `scripts/merge_ja_en_mfa.py:1-360`
- Modify: `scripts/ja_en_stage_inputs.py:1-420`
- Modify: `tests/test_ja_stage_inputs.py`
- Modify: `tests/test_ja_en_seams.py`

**Interfaces:**
- Consumes: semantic v2 graphs and raw Japanese/English MFA interval records.
- Produces: `bind_native_phone_graph(alignment: Mapping[str, Any], semantic_graphs: Sequence[Mapping[str, Any]]) -> dict[str, Any]` and `ja-en-alignment-v3` with `native_phones` in chronological order.

- [ ] **Step 1: Add repeated-label and wrong-alias regressions**

```python
def test_repeated_native_labels_join_by_interval_and_order():
    alignment = merged_fixture(phones=[
        phone("p0", "a", "ju_000001", raw=4),
        phone("p1", "a", "ju_000001", raw=5),
        phone("p2", "a", "ju_000002", raw=1),
    ])
    bound = bind_native_phone_graph(alignment, semantic_graphs_for_repeated_a())
    assert [row["phone_id"] for row in bound["native_phones"]] == ["p0", "p1", "p2"]
    assert [row["raw_interval_id"] for row in bound["native_phones"]] == [4, 5, 1]
    assert bound["native_phones"][2]["alias"] == "ju_000002"

def test_alias_mismatch_never_falls_back_to_all_actual_phones():
    with pytest.raises(JAContractError, match="native_basic_mapping_ambiguous"):
        bind_native_phone_graph(
            merged_fixture(phones=[phone("p0", "a", "ju_wrong", raw=1)]),
            semantic_graphs_for_alias("ju_expected"),
        )
```

- [ ] **Step 2: Run the stage-input tests and observe the positional fallback**

Run: `pytest -q tests/test_ja_stage_inputs.py tests/test_ja_en_seams.py`

Expected: FAIL because `_enrich_merged_alignment` can substitute all actual phones after a missed token join.

- [ ] **Step 3: Replace fallback matching with an explicit composite join**

```python
def bind_native_phone_graph(alignment, semantic_graphs):
    graph_by_key = {}
    for graph in semantic_graphs:
        for node in graph["native_phone_templates"]:
            key = (graph["uid"], node["token_id"], node["alias"])
            graph_by_key.setdefault(key, []).append((graph, node))
    native = []
    occurrence_index = {}
    for phone in alignment["native_phones"]:
        key0 = (alignment["uid"], phone["token_id"], phone["alias"])
        position = occurrence_index.get(key0, 0)
        candidates = graph_by_key.get(key0, [])
        if position >= len(candidates):
            raise JAContractError(
                "native_basic_mapping_ambiguous", "no ordered semantic template", "$.native_phones"
            )
        graph, template = candidates[position]
        occurrence_index[key0] = position + 1
        native.append(_bind_one_native(phone, graph, template))
    return {**dict(alignment), "native_phones": native}
```

`_bind_one_native` additionally checks native label, selected dictionary pronunciation position, and raw interval provenance before copying ordered `basic_phone_ids`, `mora_ids`, and `transform`.

- [ ] **Step 4: Version merge output and assert immutable integer boundaries**

```python
def test_alignment_v3_retains_raw_mfa_boundaries():
    row = handle_merge_fixture()
    assert row["schema"] == "ja-en-alignment-v3"
    for phone, raw in zip(row["native_phones"], row["raw_mfa"]["phones"], strict=True):
        assert (phone["start_sample"], phone["end_sample"]) == (
            raw["start_sample"], raw["end_sample"]
        )
        assert phone["boundary_source"] == "mfa_native_interval"
        assert phone["source_axis"] == raw["source_axis"]
        assert phone["alignment_axis"] == raw["alignment_axis"]
        assert phone["training_axis"] == raw["training_axis"]
        assert phone["run_id"] == raw["run_id"]
```

Run: `pytest -q tests/test_ja_stage_inputs.py tests/test_ja_en_seams.py tests/test_ja_locked_aliases.py`

Expected: PASS.

- [ ] **Step 5: Commit the authoritative native-phone join**

```bash
git add scripts/merge_ja_en_mfa.py scripts/ja_en_stage_inputs.py tests/test_ja_stage_inputs.py tests/test_ja_en_seams.py tests/test_ja_locked_aliases.py
git commit -m "feat: bind alignment phones to semantic graph"
```

## Task 5: Implement pure mora-tone resolution and native-phone projection

**Files:**
- Create: `scripts/ja_prosody.py`
- Create: `tests/test_ja_prosody_projection.py`
- Modify: `tests/test_ja_prosody_schema.py`

**Interfaces:**
- Consumes: semantic v2 graph, alignment v3, frontend accent evidence, optional parsed override and lexicon resources.
- Produces: `mora_tones_from_phrase(mora_count: int, nucleus: int) -> list[str]`; `resolve_mora_tones(graph: Mapping[str, Any], frontend: Mapping[str, Any], manual_overrides: Mapping[str, Any] | None, accent_lexicon: Mapping[str, Any] | None) -> list[dict[str, Any]]`; `project_native_phone_tones(alignment: Mapping[str, Any], graph: Mapping[str, Any], mora_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]`; `build_prosody_alignment(...) -> dict[str, Any]`.

- [ ] **Step 1: Write tone-shape, precedence, and projection tests**

```python
@pytest.mark.parametrize(("count", "nucleus", "tones"), [
    (3, 0, ["L", "H", "H"]),
    (3, 1, ["H", "L", "L"]),
    (4, 2, ["L", "H", "L", "L"]),
    (4, 3, ["L", "H", "H", "L"]),
])
def test_mora_tones_from_phrase(count, nucleus, tones):
    assert mora_tones_from_phrase(count, nucleus) == tones

def test_source_priority_is_not_voting(graph, frontend):
    rows = resolve_mora_tones(
        graph, frontend,
        manual_overrides={"locked_reading_digest": graph["locked_reading_digest"], "tones": ["H", "L"]},
        accent_lexicon={"tones": ["L", "H"]},
    )
    assert [row["tone"] for row in rows] == ["H", "L"]
    assert {row["tone_source"] for row in rows} == {"manual_override"}
    assert all(row["overridden_sources"] for row in rows)

def test_long_vowel_projects_vector_without_splitting_interval():
    result = build_prosody_alignment(long_o_alignment(), graph_for("コー"), valid_frontend("コー"))
    phone = next(row for row in result["native_phones"] if row["native_phone"] == "oː")
    assert phone["phone_kana"] == "コ|ー"
    assert phone["phone_tone"] == "H|L"
    assert (phone["start_sample"], phone["end_sample"]) == (160, 480)
    assert phone["duration_group"]["internal_boundaries_known"] is False
```

- [ ] **Step 2: Run the new unit suite and verify it fails at import**

Run: `pytest -q tests/test_ja_prosody_projection.py tests/test_ja_prosody_schema.py`

Expected: FAIL because `scripts.ja_prosody` does not exist.

- [ ] **Step 3: Implement deterministic phrase tone and source precedence**

```python
TONE_VALUES = frozenset({"H", "L", "UNK"})

def mora_tones_from_phrase(mora_count, nucleus):
    if mora_count < 1 or nucleus < 0 or nucleus > mora_count:
        raise JAContractError("accent_phrase_unresolved", "invalid phrase nucleus")
    if nucleus == 1:
        return ["H"] + ["L"] * (mora_count - 1)
    tones = ["L"] + ["H"] * (mora_count - 1)
    if 1 < nucleus < mora_count:
        for index in range(nucleus, mora_count):
            tones[index] = "L"
    return tones

def _choose_source(graph, frontend, manual_overrides, accent_lexicon):
    digest = graph["locked_reading_digest"]
    for name, resource in (
        ("manual_override", manual_overrides),
        ("fixed_accent_lexicon", accent_lexicon),
    ):
        if resource and resource.get("locked_reading_digest") == digest:
            return name, resource
    if frontend.get("accent_evidence_valid") and frontend.get("locked_reading_digest") == digest:
        return "contextual_frontend_prediction", frontend["accent_evidence"]
    return "unknown", None
```

Known rows store resource path/entry/provider revision, resource SHA-256, locked-reading digest, adapter version, and evidence digest. If any is absent, raise `tone_provenance_missing` instead of emitting known tone.

- [ ] **Step 4: Implement ordered projection and special values**

```python
def _display_for_native(phone, mora_by_id):
    if phone["language"] != "ja":
        return "", "NA"
    ordered = [mora_by_id[mora_id] for mora_id in phone["mora_ids"]]
    if not ordered:
        raise JAContractError("phone_tone_projection_lossy", "Japanese phone has no mora")
    return (
        "|".join(row["kana"] for row in ordered),
        "|".join(row["tone"] for row in ordered),
    )

def project_native_phone_tones(alignment, graph, mora_rows):
    mora_by_id = {row["mora_id"]: row for row in mora_rows}
    result = []
    for phone in alignment["native_phones"]:
        kana, tone = _display_for_native(phone, mora_by_id)
        result.append({**dict(phone), "phone_kana": kana, "phone_tone": tone})
    return result
```

Add assertions for `キット`, `オンナ`, `ウンメー`, `グッズ`, devoicing, elision, final `ッ`, mixed JA/EN, pure English, and all-`UNK`. Explicitly assert that `f0_observed=False` leaves known textual tone unchanged.

- [ ] **Step 5: Test negative cardinality and tampering cases**

```python
def test_projection_rejects_conflicting_single_value_for_cross_mora_phone():
    phone = long_o_alignment()["native_phones"][1]
    phone["phone_tone"] = "H"
    with pytest.raises(JAContractError, match="phone_tone_projection_lossy"):
        validate_projected_phone(phone, moras_for_long_o())

def test_english_never_receives_japanese_tone():
    result = build_prosody_alignment(english_alignment(), empty_ja_graph(), empty_frontend())
    assert result["moras"] == []
    assert result["basic_phones"] == []
    assert {row["phone_tone"] for row in result["native_phones"]} == {"NA"}

def test_silence_breath_and_laughter_have_no_mora_and_use_na():
    result = build_prosody_alignment(event_alignment(["sil", "breath", "laugh"]), empty_ja_graph(), empty_frontend())
    assert result["moras"] == []
    assert all(row["mora_ids"] == [] for row in result["native_phones"])
    assert all(row["phone_tone"] == "NA" for row in result["native_phones"])

def test_phrase_mora_cardinality_mismatch_is_rejected():
    graph = graph_for("サクラ")
    frontend = valid_frontend("サクラ")
    frontend["accent_evidence"]["accent_phrases"][0]["mora_count"] = 2
    with pytest.raises(JAContractError, match="tone_cardinality_mismatch"):
        build_prosody_alignment(sakura_alignment(), graph, frontend)
```

Run: `pytest -q tests/test_ja_prosody_projection.py tests/test_ja_prosody_schema.py`

Expected: PASS.

- [ ] **Step 6: Commit the pure prosody core**

```bash
git add scripts/ja_prosody.py tests/test_ja_prosody_projection.py tests/test_ja_prosody_schema.py
git commit -m "feat: project mora tones onto MFA phones"
```

## Task 6: Wire the prosody stage, inputs, and cache identity

**Files:**
- Modify: `scripts/ja_prosody.py`
- Modify: `scripts/ja_en_stage_inputs.py:420-980`
- Modify: `scripts/run_ja_en_pipeline.py:55-110,230-260,410-455,760-900,1138-1170`
- Modify: `tests/test_ja_stage_inputs.py`
- Modify: `tests/test_ja_resume_identity.py`

**Interfaces:**
- Consumes: `build_prosody_alignment` from Task 5 and workspace artifacts from reading, frontend, semantic, and merge stages.
- Produces: `assemble_prosody_rows(config: Mapping[str, Any], workspace: Path, merged_alignments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]`; `handle_prosody(config: Mapping[str, Any], stage_dir: Path) -> StageResult`; revised `assemble_tts_rows(...)` consuming only `ja-prosody-alignment-v1` rows.

- [ ] **Step 1: Add an end-to-end stage-input test with duplicate labels**

```python
def test_assemble_prosody_rows_joins_by_uid_token_and_alias(tmp_path):
    workspace = staged_workspace(tmp_path, repeated_phone_labels=True)
    rows = assemble_prosody_rows(config_for(workspace), workspace, load_merge_rows(workspace))
    assert [row["uid"] for row in rows] == ["mixed-1"]
    assert rows[0]["schema"] == "ja-prosody-alignment-v1"
    assert [row["raw_interval_id"] for row in rows[0]["native_phones"]] == [1, 2, 3, 4]
    assert rows[0]["native_phones"][2]["phone_tone"] == "NA"
```

- [ ] **Step 2: Run orchestration tests and observe that no prosody handler exists**

Run: `pytest -q tests/test_ja_stage_inputs.py tests/test_ja_resume_identity.py`

Expected: FAIL because the registry/autoloader and input bridge skip `prosody`.

- [ ] **Step 3: Add stage handler and ledger-preserving per-UID failure behavior**

```python
def handle_prosody(config, stage_dir):
    workspace = stage_dir.parent.parent
    receipt_path = stage_dir / "receipt.json"
    source = Path(config["prosody"]["alignment_jsonl"])
    source_rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, failures = [], []
    for alignment in source_rows:
        try:
            rows.extend(assemble_prosody_rows(config, workspace, [alignment]))
        except JAContractError as error:
            failures.append({"uid": alignment["uid"], **error.as_dict()})
    output = stage_dir / "prosody_alignments.jsonl"
    atomic_write_bytes(output, b"".join(canonical_json(row) for row in rows))
    status = "COMPLETE" if not failures else "PARTIAL"
    receipt = make_receipt(
        stage="prosody", status=status,
        inputs={"artifacts": [{"path": str(source), "sha256": sha256_file(source)}]},
        outputs=[output], params={"implementation": "ja-mora-tone-v1"},
        errors=failures,
    )
    atomic_write_json(receipt_path, receipt, workspace=workspace)
    return StageResult(stage="prosody", status=status, receipt_path=str(receipt_path))

def register_stages(registrar):
    registrar("prosody", handle_prosody, output_namespace="prosody")
```

Use existing atomic JSONL and receipt helpers rather than duplicate writers. Keep every failed UID in unresolved/rejected ledger; never silently drop it.

- [ ] **Step 4: Register prosody and make TTS consume its artifact**

```python
def _autoload_stages():
    modules = (
        "ja_audio", "ja_asr_crossval", "ja_frontend", "ja_phone_adapter",
        "ja_en_anchors", "align_japanese_mfa", "merge_ja_en_mfa",
        "ja_prosody", "run_julius_diagnostic", "ja_tts_export",
    )
    for name in modules:
        module = importlib.import_module(f"scripts.{name}")
        registrar = getattr(module, "register_stages", None)
        if callable(registrar):
            registrar(register_stage)
```

In `_prepare_stage_inputs`, set `prosody.alignment_jsonl` from merge and set `tts.alignment_jsonl` from `prosody/prosody_alignments.jsonl`. Add `scripts/ja_prosody.py` to implementation identity.

Add a non-destructive workspace override for acceptance runs: `build_parser()` accepts `--workspace`, and `main()` replaces only validated `config["workspace"]` before preflight. Reject a non-empty workspace unless `--resume` is supplied; never delete it.

```python
parser.add_argument("--workspace", help="override config workspace; must be new/empty unless --resume")
# after loading YAML, before preflight
if args.workspace:
    config["workspace"] = str(Path(args.workspace).expanduser().absolute())
```

- [ ] **Step 5: Prove cache invalidation stops at the correct boundary**

```python
def test_tone_override_change_invalidates_only_prosody_downstream(workspace):
    before = stage_identities(workspace)
    mutate_tone_override(workspace, ["H", "L"])
    after = planned_stage_identities(workspace)
    assert unchanged(before, after, through="merge")
    assert changed(before, after, stages=("prosody", "tts", "verify"))

def test_mfa_dictionary_change_invalidates_align_and_all_downstream(workspace):
    before = stage_identities(workspace)
    mutate_dictionary(workspace)
    after = planned_stage_identities(workspace)
    assert changed(before, after, stages=("align", "merge", "prosody", "tts", "verify"))

def test_workspace_override_never_reuses_nonempty_directory_without_resume(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "evidence.txt").write_text("keep", encoding="utf-8")
    assert main(["--config", str(config_path(tmp_path)), "--workspace", str(occupied)]) == 2
    assert (occupied / "evidence.txt").read_text(encoding="utf-8") == "keep"
```

Run: `pytest -q tests/test_ja_stage_inputs.py tests/test_ja_resume_identity.py`

Expected: PASS.

- [ ] **Step 6: Commit the stage integration**

```bash
git add scripts/ja_prosody.py scripts/ja_en_stage_inputs.py scripts/run_ja_en_pipeline.py tests/test_ja_stage_inputs.py tests/test_ja_resume_identity.py
git commit -m "feat: wire prosody pipeline stage"
```

## Task 7: Export TTS v2 and an exact five-tier continuous TextGrid

**Files:**
- Modify: `scripts/ja_tts_export.py:1-430`
- Create: `tests/test_ja_five_track_textgrid.py`
- Modify: `tests/test_ja_tts_export.py`
- Modify: `tests/test_boundary_punctuation_display_regressions.py`

**Interfaces:**
- Consumes: one `ja-prosody-alignment-v1` row plus authoritative audio receipt.
- Produces: `build_training_record(...) -> dict[str, Any]` with schema v2; `build_five_track_tiers(record: Mapping[str, Any]) -> list[dict[str, Any]]`; `render_textgrid(record: Mapping[str, Any]) -> str`; `export_tts_artifacts(...) -> dict[str, Path]`.

- [ ] **Step 1: Replace the three-tier test with exact names, ordering, and vector labels**

```python
def test_export_writes_exact_five_track_textgrid(tmp_path):
    record = training_record_v2_fixture(tmp_path, text=' 「コー」 game ')
    paths = export_tts_artifacts(record, tmp_path / "out")
    parsed = parse_textgrid(paths["textgrid"])
    assert [tier["name"] for tier in parsed["tiers"]] == [
        "original_text", "kana", "mfa_phone", "phone_kana", "phone_tone"
    ]
    assert labels(parsed, "phone_kana") == ["コ", "コ|ー", "", ""]
    assert labels(parsed, "phone_tone") == ["H", "H|L", "NA", "NA"]
    for tier_name in ("mfa_phone", "phone_kana", "phone_tone"):
        assert boundaries(parsed, tier_name) == boundaries(parsed, "mfa_phone")
```

- [ ] **Step 2: Add continuous coverage and escaping regressions**

```python
def test_writer_preserves_whitespace_punctuation_quotes_and_one_sample_boundaries(tmp_path):
    record = training_record_v2_fixture(
        tmp_path, text='  「声"」!  ', phone_bounds=[(0, 1), (1, 2), (2, 16000)]
    )
    grid = parse_textgrid_text(render_textgrid(record))
    assert reconstruct_display(grid, record["display_attachments"]) == '  「声"」!  '
    assert all(tier_is_continuous(tier, 0, 16000) for tier in grid["tiers"])
    assert boundaries(grid, "mfa_phone")[:2] == [(0, 1), (1, 2)]
    assert labels(grid, "original_text")[0].endswith('「声"」!')

def test_empty_coverage_intervals_are_not_authoritative_phones(tmp_path):
    grid = parse_textgrid_text(render_textgrid(training_record_with_initial_gap(tmp_path)))
    assert len(authoritative_segments(grid, "mfa_phone")) == 2
    assert labels(grid, "mfa_phone")[0] == ""
```

- [ ] **Step 3: Run exporter tests and confirm old tier names fail**

Run: `pytest -q tests/test_ja_tts_export.py tests/test_ja_five_track_textgrid.py tests/test_boundary_punctuation_display_regressions.py`

Expected: FAIL because the current writer emits `words`, `phones`, and `language`.

- [ ] **Step 4: Build lexical and phone tiers from authoritative segment objects**

```python
TIER_NAMES = (
    "original_text", "kana", "mfa_phone", "phone_kana", "phone_tone"
)

def build_five_track_tiers(record):
    words = record["words"]
    phones = record["native_phones"]
    return [
        _continuous_tier("original_text", [_word_interval(w, w["source_text"]) for w in words], record),
        _continuous_tier("kana", [_word_interval(w, w["kana"] if w["language"] == "ja" else "") for w in words], record),
        _continuous_tier("mfa_phone", [_phone_interval(p, _native_label(p)) for p in phones], record),
        _continuous_tier("phone_kana", [_phone_interval(p, p["phone_kana"]) for p in phones], record),
        _continuous_tier("phone_tone", [_phone_interval(p, p["phone_tone"]) for p in phones], record),
    ]

def _native_label(phone):
    return "" if phone.get("is_silence") else f'{phone["language"]}:{phone["native_phone"]}'
```

`_continuous_tier` inserts display-only empty intervals for uncovered spans, stores no synthetic segment ID, rejects overlap, doubles embedded double quotes for Praat, and converts samples to decimal seconds only when serializing.

- [ ] **Step 5: Emit v2 records with group durations and separate tone/F0 masks**

```python
record = {
    "schema": "tts-training-record-v2",
    "textgrid_schema": "five-track-textgrid-v1",
    "uid": alignment["uid"],
    "sample_rate": sample_rate,
    "frame_count": frame_count,
    "train_wav": str(train_wav),
    "alignment_wav": str(alignment_wav),
    "words": alignment["words"],
    "native_phones": alignment["native_phones"],
    "moras": alignment["moras"],
    "basic_phones": alignment["basic_phones"],
    "duration_groups": alignment["duration_groups"],
    "display_attachments": alignment["display_attachments"],
    "quality_masks": build_quality_masks(alignment),
}
```

Each multi-basic duration group records only `total_duration_samples`, `internal_boundaries_known=false`, `boundary_source="unknown_inside_mfa_interval"`, and `duration_loss_mode="group_sum"`; no per-basic duration is fabricated.

- [ ] **Step 6: Run exporter regressions and commit**

Run: `pytest -q tests/test_ja_tts_export.py tests/test_ja_five_track_textgrid.py tests/test_boundary_punctuation_display_regressions.py`

Expected: PASS, including pure-English `NA` and Unicode quote round-trip cases.

```bash
git add scripts/ja_tts_export.py tests/test_ja_tts_export.py tests/test_ja_five_track_textgrid.py tests/test_boundary_punctuation_display_regressions.py
git commit -m "feat: export five-track TTS artifacts"
```

## Task 8: Independently verify structure, provenance, tampering, and release policy

**Files:**
- Modify: `scripts/verify_ja_en_tts.py:1-1320`
- Modify: `tests/test_ja_independent_verifier.py`
- Modify: `tests/test_ja_canary_gate.py`

**Interfaces:**
- Consumes: files reopened from receipts: v2 JSONL, five-track TextGrid, raw MFA TextGrid, locked dictionary/aliases, semantic v2 graphs, audio receipts, and stage identities.
- Produces: `verify_training_record(record: Mapping[str, Any], *, external: Mapping[str, Path]) -> list[dict[str, Any]]`; `verify_five_track_textgrid(...) -> list[dict[str, Any]]`; release result honoring `publish.require_all_tones`.

- [ ] **Step 1: Add one-sample boundary, label, order, provenance, and English tamper tests**

```python
@pytest.mark.parametrize(("mutation", "code"), [
    (lambda bundle: shift_tier_boundary(bundle, "phone_tone", index=1, samples=1), "five_track_boundary_mismatch"),
    (lambda bundle: replace_label(bundle, "phone_kana", index=1, value="ー|コ"), "phone_tone_projection_lossy"),
    (lambda bundle: replace_label(bundle, "phone_tone", index=1, value="H"), "phone_tone_projection_lossy"),
    (lambda bundle: drop_basic_role(bundle, "bp1"), "native_basic_mapping_ambiguous"),
    (lambda bundle: change_alias(bundle, "p1", "ju_other"), "verifier_failed"),
    (lambda bundle: change_raw_interval(bundle, "p1", 99), "verifier_failed"),
    (lambda bundle: set_english_tone(bundle, "p_en", "H"), "phone_tone_projection_lossy"),
    (lambda bundle: drop_tone_resource(bundle, "m0"), "tone_provenance_missing"),
    (lambda bundle: mark_estimated_boundary_as_mfa(bundle, "dg0"), "verifier_failed"),
])
def test_external_tampering_is_rejected(tmp_path, mutation, code):
    bundle = accepted_bundle(tmp_path)
    mutation(bundle)
    errors = verify_workspace_bundle(bundle.root)
    assert code in {row["code"] for row in errors}
```

- [ ] **Step 2: Run the independent verifier suite and confirm v1 assumptions fail**

Run: `pytest -q tests/test_ja_independent_verifier.py tests/test_ja_canary_gate.py`

Expected: FAIL because the verifier expects old schemas and three tiers.

- [ ] **Step 3: Recompute every display value and relation from reopened JSON**

```python
def _expected_phone_display(phone, mora_by_id):
    if phone["language"] != "ja":
        return "", "NA"
    ordered = [mora_by_id[key] for key in phone["mora_ids"]]
    return (
        "|".join(row["kana"] for row in ordered),
        "|".join(row["tone"] for row in ordered),
    )

def _verify_phone_tiers(record, parsed_grid):
    authority = authoritative_intervals(parsed_grid, "mfa_phone")
    kana = authoritative_intervals(parsed_grid, "phone_kana")
    tone = authoritative_intervals(parsed_grid, "phone_tone")
    if [bounds(row) for row in authority] != [bounds(row) for row in kana] \
            or [bounds(row) for row in authority] != [bounds(row) for row in tone]:
        raise JAContractError("five_track_boundary_mismatch", "phone tiers differ")
    mora_by_id = {row["mora_id"]: row for row in record["moras"]}
    for phone, kana_interval, tone_interval in zip(record["native_phones"], kana, tone, strict=True):
        expected_kana, expected_tone = _expected_phone_display(phone, mora_by_id)
        if (kana_interval["text"], tone_interval["text"]) != (expected_kana, expected_tone):
            raise JAContractError("phone_tone_projection_lossy", "display projection differs")
```

The verifier must parse raw MFA TextGrid independently and compare interval IDs/bounds/labels, reopen locked pronunciations and aliases, recompute file SHA-256 values, and never trust producer summary booleans.

For every duration group, reject `internal_boundaries_known=true` unless a permitted estimator identity, confidence, and `boundary_source="estimated"` are present; always reject estimated internal boundaries labeled `mfa_native_interval`.

- [ ] **Step 4: Enforce graph coverage and release-only known-tone policy**

```python
def _verify_graph_coverage(record, require_all_tones):
    mora_ids = {row["mora_id"] for row in record["moras"]}
    basic_ids = {row["basic_phone_id"] for row in record["basic_phones"]}
    if any(row["mora_id"] not in mora_ids for row in record["basic_phones"]):
        raise JAContractError("native_basic_mapping_ambiguous", "basic phone has invalid mora")
    covered = {key for phone in record["native_phones"] for key in phone["basic_phone_ids"]}
    elided = {row["basic_phone_id"] for row in record["basic_phones"] if row["realization"] == "elided"}
    if covered | elided != basic_ids:
        raise JAContractError("native_basic_mapping_ambiguous", "basic phone coverage differs")
    if require_all_tones and any(row["tone"] == "UNK" for row in record["moras"]):
        raise JAContractError("publish_blocked", "Japanese mora tone is unknown")
```

Add a test that hashes raw MFA outputs before and after the rejected release attempt and asserts equality. Development verification of the same all-`UNK` bundle remains structurally valid.

- [ ] **Step 5: Run verifier and tamper regressions and commit**

Run: `pytest -q tests/test_ja_independent_verifier.py tests/test_ja_canary_gate.py tests/test_ja_five_track_textgrid.py`

Expected: PASS.

```bash
git add scripts/verify_ja_en_tts.py tests/test_ja_independent_verifier.py tests/test_ja_canary_gate.py tests/test_ja_five_track_textgrid.py
git commit -m "feat: verify five-track prosody artifacts"
```

## Task 9: Document and run the fresh-workspace acceptance matrix

**Files:**
- Modify: `docs/JA_EN_PIPELINE.md`
- Modify: `README.md`
- Modify: `tests/test_ja_resume_identity.py`
- Modify: `tests/test_julius_diagnostic_isolation.py`
- Modify: `tests/test_ja_tts_export.py`

**Interfaces:**
- Consumes: all production interfaces from Tasks 1-8.
- Produces: user-facing stage/artifact documentation and a reproducible fresh-workspace acceptance command sequence.

- [ ] **Step 1: Add integration assertions for fresh run, resume, override invalidation, and Julius isolation**

```python
def test_fresh_mixed_run_resumes_without_hash_drift(fresh_pipeline):
    first = fresh_pipeline.run()
    assert first["integrity_ok"] is True
    assert first["release_ready"] is False
    hashes = output_hashes(first.workspace)
    second = fresh_pipeline.resume()
    assert output_hashes(second.workspace) == hashes
    assert second.recomputed_stages == []

def test_julius_toggle_does_not_change_production_artifacts(fresh_pipeline):
    without = fresh_pipeline.run(julius=False)
    with_diag = fresh_pipeline.run(julius=True)
    assert hash_named(without.workspace, "tts_training_records.jsonl") == hash_named(with_diag.workspace, "tts_training_records.jsonl")
    assert hash_glob(without.workspace, "*.TextGrid") == hash_glob(with_diag.workspace, "*.TextGrid")
```

- [ ] **Step 2: Update the operator documentation with exact contracts**

Document:

```text
merge/merged_alignments.jsonl     ja-en-alignment-v3
prosody/prosody_alignments.jsonl ja-prosody-alignment-v1
tts/tts_training_records.jsonl   tts-training-record-v2
tts/<uid>.TextGrid               five-track-textgrid-v1

TextGrid tiers, in order:
original_text, kana, mfa_phone, phone_kana, phone_tone
```

Also document source priority, `UNK` versus `NA`, vector labels such as `コ|ー` / `H|L`, group-only durations, reading-override invalidation, the release gate, and that a pre-v2 workspace must not be resumed as current output.

- [ ] **Step 3: Run the focused Japanese/English suite**

Run:

```bash
pytest -q \
  tests/test_ja_en_foundation.py \
  tests/test_ja_frontend_contract_w2.py \
  tests/test_ja_phone_adapter.py \
  tests/test_ja_phone_adapter_w2.py \
  tests/test_ja_mora_graph.py \
  tests/test_ja_stage_inputs.py \
  tests/test_ja_prosody_schema.py \
  tests/test_ja_prosody_projection.py \
  tests/test_ja_tts_export.py \
  tests/test_ja_five_track_textgrid.py \
  tests/test_ja_independent_verifier.py \
  tests/test_ja_resume_identity.py \
  tests/test_julius_diagnostic_isolation.py
```

Expected: PASS with no skipped contract tests.

- [ ] **Step 4: Run syntax compilation and the full repository suite**

Run:

```bash
python -m compileall -q scripts tests
pytest -q
```

Expected: both commands exit 0. Existing unrelated environmental skips remain documented by pytest; no new failure or xfail is introduced.

- [ ] **Step 5: Run a fresh pure-JA, pure-EN, and mixed CLI acceptance workspace**

Run:

```bash
python scripts/run_ja_en_pipeline.py \
  --config /home/user/ja-en-task-artifacts/final-cli-proof/config-mixed.yaml \
  --workspace "$acceptance_workspace"
python scripts/verify_ja_en_tts.py \
  --workspace "$acceptance_workspace" \
  --integrity-only
```

Before these commands, set the isolated target without deleting any existing path:

```bash
acceptance_workspace=$(mktemp -d /tmp/japmfa-five-track-acceptance.XXXXXX)
```

Expected: the run reaches verification; the configured acceptance manifest contains pure Japanese, pure English, and mixed records; `integrity_ok=true`; `release_ready=false` when production gold/license gates are intentionally absent. Inspect one mixed TextGrid and assert five tier names plus exact phone-tier sample boundaries against the JSONL record.

- [ ] **Step 6: Commit documentation and acceptance coverage**

```bash
git add README.md docs/JA_EN_PIPELINE.md tests/test_ja_resume_identity.py tests/test_julius_diagnostic_isolation.py tests/test_ja_tts_export.py
git commit -m "docs: document five-track prosody pipeline"
```

## Final branch verification

- [ ] Run `git diff --check` and require no whitespace errors.
- [ ] Run `git status --short` and distinguish pre-existing user changes from files changed by this plan.
- [ ] Re-run the focused suite from Task 9 after the final commit.
- [ ] Re-run `python -m compileall -q scripts tests`.
- [ ] Record the fresh acceptance workspace receipt path and hashes in the implementation handoff; do not commit generated audio, model files, workspaces, or caches.
