import json
import wave
from pathlib import Path

from scripts.ja_tts_export import build_training_record, export_tts_artifacts
from scripts.verify_ja_en_tts import verify_workspace
from scripts.ja_en_schema import artifact_record
from scripts.ja_en_schema import make_receipt
from scripts.ja_audio import make_audio_receipt
from scripts.ja_audio import inspect_wav, prepare_alignment_wav
from scripts.verify_ja_en_tts import _textgrid_parse, verify_training_record


def _record(tmp_path: Path):
    train = tmp_path / "train.wav"; align = tmp_path / "align.wav"
    import wave
    for p in (train, align):
        with wave.open(str(p), "wb") as out:
            out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes(b"\0\0" * 16000)
    alignment = {"schema": "ja-en-alignment-v2", "uid": "u1", "words": [{"unit_id": "w", "text": "さ", "language": "ja", "start_sample": 0, "end_sample": 16000}], "phones": [{"unit_id": "w", "phone_id": "p", "phone": "a", "native_phone": "a", "language": "ja", "start_sample": 0, "end_sample": 16000, "raw_interval_id": 1, "alias": "ju_000000", "mora_ids": ["m"]}], "languages": [{"language": "ja", "unit_ids": ["w"]}], "mora_graph": {"moras": [{"mora_id": "m", "phone_ids": ["p"]}], "relations": [{"mora_id": "m", "phone_id": "p"}]}}
    alignment.update({"selected_reading": "さ", "locked_aliases": [{"alias": "ju_000000", "pronunciation": ["a"]}], "native_inventory": {"ja": ["a"], "en": []}, "raw_mfa": {"textgrid_path": str(tmp_path / "out" / "u1.TextGrid"), "phones": [{"phone_id": "p", "raw_interval_id": 1, "unit_id": "w"}]}, "reading_evidence": {"selected_reading": "さ", "status": "manual_verified"}, "partition": {"verified": ["u1"], "rejected": [], "unresolved": []}})
    out = tmp_path / "out"
    receipt_audio = make_audio_receipt("u1", train, train, align, alignment_transform={"method": "identity_fixture_v1", "source_start": 0, "source_end": 16000, "output_start": 0, "output_frames": 16000})
    export_tts_artifacts(build_training_record(alignment, train_wav=train, alignment_wav=align, speaker="spk", audio_receipt=receipt_audio), out)
    row_path = out / "tts_training_records.jsonl"
    row = json.loads(row_path.read_text())
    # Bind the synthetic record to independent authority artifacts.  These
    # files model the W3 contract and keep the fixture useful for tamper
    # tests without treating producer JSON as its own evidence.
    authority = out / "authority"; authority.mkdir(parents=True, exist_ok=True)
    raw_grid = authority / "u1.raw.TextGrid"
    raw_grid.write_text('''File type = "ooTextFile"\nObject class = "TextGrid"\nxmin = 0\nxmax = 1\ntiers? <exists>\nsize = 2\nitem []:\n    item [1]:\n        class = "IntervalTier"\n        name = "words"\n        xmin = 0\n        xmax = 1\n        intervals: size = 1\n        intervals [1]:\n            xmin = 0\n            xmax = 1\n            text = "ju_000000"\n    item [2]:\n        class = "IntervalTier"\n        name = "phones"\n        xmin = 0\n        xmax = 1\n        intervals: size = 1\n        intervals [1]:\n            xmin = 0\n            xmax = 1\n            text = "a"\n''', encoding="utf-8")
    dictionary = authority / "u1.dict"; dictionary.write_text("ju_000000 a\n", encoding="utf-8")
    archive = authority / "u1.zip"
    import zipfile
    with zipfile.ZipFile(archive, "w") as zf: zf.writestr("metadata.json", json.dumps({"phones": ["a"]}))
    from scripts.ja_en_schema import sha256_file
    raw_hash, dict_hash, archive_hash = (sha256_file(path) for path in (raw_grid, dictionary, archive))
    semantic = authority / "semantic.json"
    semantic.write_text(json.dumps({"graphs": [{"uid": "u1", "token_id": "w", "candidate_id": "cand-0", "semantic_phone_nodes": [{"id": "sp", "kind": "semantic_phone", "phone": "a", "mora_ids": ["m"]}], "mora_nodes": [{"id": "m", "kind": "mora"}], "nodes": [{"id": "sp"}, {"id": "m"}], "edges": [{"relation": "mora_phone", "mora_id": "m", "phone_id": "sp"}], "target": {"phones": ["a"]}}]}), encoding="utf-8")
    semantic_hash = sha256_file(semantic)
    row["partition"] = {"expected_unit_ids": ["w"], "verified": ["w"], "rejected": [], "unresolved": []}
    row["semantic_graph_path"] = str(semantic); row["semantic_graph_sha256"] = semantic_hash
    row["phones"][0].update(run_id="run-u1", raw_interval_index=1, raw_artifact_path=str(raw_grid), raw_artifact_sha256=raw_hash)
    row["raw_mfa"] = {"runs": [{"run_id": "run-u1", "language": "ja", "unit_ids": ["w"], "raw_textgrid": {"path": str(raw_grid), "sha256": raw_hash}, "raw_tier": "phones", "sample_rate": 16000, "crop_offset_sample": 0, "ownership_start_sample": 0, "ownership_end_sample": 16000, "locked_dict_path": str(dictionary), "locked_dict_sha256": dict_hash, "native_inventory_path": str(archive), "native_inventory_sha256": archive_hash, "aliases": row["locked_aliases"]}], "model_assets": {}}
    row_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (out / "alias_map.jsonl").write_text(json.dumps({"uid": "u1", "token_id": "w", "candidate_id": "cand-0", "alias": "ju_000000", "pronunciation": ["a"]}) + "\n", encoding="utf-8")
    (out / "supply_chain_lock.json").write_text(json.dumps({"schema": "ja-supply-chain-lock-v1", "resources": [{"id": "fixture", "kind": "diagnostic", "source_url": "fixture://offline", "revision": "fixture", "license": {"status": "reviewed"}, "artifacts": [], "artifact_hashes": []}]}), encoding="utf-8")
    rows = [{"id": f"fixture-{i}", "accepted": True, "bucket": ("ja_to_en", "en_to_ja", "no_pause", "short_english")[i // 10], "predicted_seam_sample": 320} for i in range(40)]
    gold = {"target_id": "gate-fixture", "sample_rate": 16000, "rows": [{"id": row["id"], "bucket": row["bucket"], "gold_seam_sample": 0} for row in rows]}
    (out / "gold.json").write_text(json.dumps(gold), encoding="utf-8")
    (out / "canary_gate.json").write_text(json.dumps({"status": "PASS", "target_id": "gate-fixture", "proof": True, "rows": rows, "gold": gold}), encoding="utf-8")
    receipt = make_receipt(stage="verify", status="COMPLETE", outputs=[out / "tts_training_records.jsonl", out / "u1.TextGrid", out / "supply_chain_lock.json", out / "gold.json", out / "canary_gate.json", out / "alias_map.jsonl", raw_grid, dictionary, archive, semantic])
    (out / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    stage = out / "stages" / "tts"; stage.mkdir(parents=True)
    (stage / "receipt.json").write_text(json.dumps(make_receipt(stage="tts", status="COMPLETE", outputs=[out / "tts_training_records.jsonl", out / "u1.TextGrid"])), encoding="utf-8")
    return out


def test_verifier_recomputes_semantic_relation_and_rejects_mutation(tmp_path: Path):
    out = _record(tmp_path)
    good = verify_workspace(out)
    assert good["ok"] is True
    path = out / "tts_training_records.jsonl"
    payload = json.loads(path.read_text())
    payload["mora_graph"]["relations"][0]["phone_id"] = "missing"
    path.write_text(json.dumps(payload) + "\n")
    bad = verify_workspace(out)
    assert bad["ok"] is False
    assert any("mora" in e["message"] for e in bad["errors"])


def test_verifier_recomputes_receipt_hash_before_accepting_semantic_mutation(tmp_path: Path):
    out = _record(tmp_path)
    outputs = [artifact_record(out / "tts_training_records.jsonl"), artifact_record(out / "u1.TextGrid")]
    (out / "receipt.json").write_text(json.dumps({"status": "PARTIAL", "outputs": outputs}), encoding="utf-8")
    payload = json.loads((out / "tts_training_records.jsonl").read_text())
    payload["phones"][0]["mora_ids"] = ["wrong"]
    (out / "tts_training_records.jsonl").write_text(json.dumps(payload) + "\n")
    report = verify_workspace(out)
    assert any(error["code"] == "receipt_hash_mismatch" for error in report["errors"])


def test_missing_receipt_is_release_blocker_without_false_complete(tmp_path: Path):
    out = _record(tmp_path)
    (out / "receipt.json").unlink()
    report = verify_workspace(out)
    assert report["integrity_ok"] is False
    assert report["release_ready"] is False
    assert report["status"] == "REJECTED"


def test_forged_canary_pass_and_nan_metric_are_rejected(tmp_path: Path):
    out = _record(tmp_path)
    gate = json.loads((out / "canary_gate.json").read_text())
    gate["rows"][0]["predicted_seam_sample"] = float("nan")
    (out / "canary_gate.json").write_text(json.dumps(gate), encoding="utf-8")
    report = verify_workspace(out)
    assert report["release_ready"] is False
    assert any(error["code"] == "publish_blocked" for error in report["errors"])


def test_independent_scipy_replay_accepts_non_16k_source(tmp_path: Path):
    """The verifier must replay W1's downmix/resample recipe, not reject it."""
    import numpy as np
    source = tmp_path / "source-8k.wav"; align = tmp_path / "align-16k.wav"
    with wave.open(str(source), "wb") as out:
        out.setnchannels(2); out.setsampwidth(2); out.setframerate(8000)
        signal = np.arange(800, dtype=np.int16)
        out.writeframes(np.column_stack((signal, -signal)).astype("<i2").tobytes())
    transform = prepare_alignment_wav(source, align, sample_rate=16000)
    alignment = _record(tmp_path)  # supplies a complete semantic fixture
    # Reuse the fixture's alignment fields but bind its audio axis to source.
    row = json.loads((tmp_path / "out" / "tts_training_records.jsonl").read_text())
    row["train_wav"] = inspect_wav(source); row["alignment_wav"] = inspect_wav(align); row["sample_rate"] = 16000
    row["phones"][0]["end_sample"] = 1600; row["durations"] = [1600]
    row["train_sample_rate"] = 8000; row["phones"][0]["train_start_sample"] = 0; row["phones"][0]["train_end_sample"] = 800
    row["source_sample_rate"] = 8000; row["phones"][0]["source_start_sample"] = 0; row["phones"][0]["source_end_sample"] = 800
    row["audio_transform"] = make_audio_receipt("u1", source, source, align,
        alignment_transform={**transform["sample_transform"], "method": "scipy_resample_poly_v1"},
        train_transform={"method": "pcm16_copy_v1", "source_start": 0, "source_end": 800})
    errors = verify_training_record(row, root=tmp_path)
    assert not any(error["code"] == "audio_transform_invalid" for error in errors), errors


def test_source_train_alignment_axes_cover_48k_pcm24_to_16k(tmp_path: Path):
    import numpy as np
    from scripts.ja_audio import prepare_training_wav
    source = tmp_path / "source-48k-pcm24.wav"; train = tmp_path / "train-48k-pcm16.wav"; align = tmp_path / "align-16k.wav"
    values = np.arange(48000, dtype=np.int32) % 1000
    packed = np.column_stack((values, -values)).astype(np.int32)
    raw = bytearray()
    for value in packed.reshape(-1):
        raw.extend(int(value).to_bytes(3, "little", signed=True))
    with wave.open(str(source), "wb") as out:
        out.setnchannels(2); out.setsampwidth(3); out.setframerate(48000); out.writeframes(bytes(raw))
    train_info = prepare_training_wav(source, train)
    align_info = prepare_alignment_wav(source, align, sample_rate=16000)
    base = _record(tmp_path)
    row = json.loads((tmp_path / "out" / "tts_training_records.jsonl").read_text())
    row["train_wav"] = inspect_wav(train); row["alignment_wav"] = inspect_wav(align); row["sample_rate"] = 16000; row["train_sample_rate"] = 48000; row["source_sample_rate"] = 48000
    row["phones"][0].update(end_sample=16000, train_start_sample=0, train_end_sample=48000, source_start_sample=0, source_end_sample=48000); row["durations"] = [16000]
    row["audio_transform"] = make_audio_receipt("u1", source, train, align, alignment_transform=align_info["sample_transform"], train_transform=train_info["sample_transform"])
    errors = verify_training_record(row, root=tmp_path)
    assert not any(error["code"] == "audio_transform_invalid" for error in errors), errors


def test_pure_english_empty_mora_graph_is_valid(tmp_path: Path):
    import wave
    wav = tmp_path / "en.wav"
    with wave.open(str(wav), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000); out.writeframes(b"\0\0" * 1600)
    alignment = {"schema": "ja-en-alignment-v2", "uid": "en1",
                 "words": [{"unit_id": "w", "text": "the", "language": "en", "start_sample": 0, "end_sample": 1600}],
                 "phones": [{"unit_id": "w", "phone_id": "p", "native_phone": "th", "language": "en", "start_sample": 0, "end_sample": 1600, "raw_interval_id": 1, "alias": "en_0"}],
                 "mora_graph": {"moras": [], "relations": []}, "selected_reading": "the",
                 "locked_aliases": [{"alias": "en_0", "pronunciation": ["th"]}], "native_inventory": {"ja": [], "en": ["th"]},
                 "raw_mfa": {"phones": [{"phone_id": "p", "raw_interval_id": 1, "unit_id": "w"}]},
                 "reading_evidence": {"selected_reading": "the"}, "partition": {"verified": ["en1"], "rejected": [], "unresolved": []}}
    audio = make_audio_receipt("en1", wav, wav, wav, alignment_transform={"method": "identity_fixture_v1", "source_start": 0, "source_end": 1600, "output_start": 0, "output_frames": 1600})
    record = build_training_record(alignment, train_wav=wav, alignment_wav=wav, audio_receipt=audio)
    errors = verify_training_record(record, root=tmp_path)
    assert not any(error["code"] == "mora_phone_relation_unresolved" for error in errors), errors


def test_rehashed_phone_mutation_still_fails_against_textgrid(tmp_path: Path):
    out = _record(tmp_path)
    jsonl = out / "tts_training_records.jsonl"
    row = json.loads(jsonl.read_text())
    row["phones"][0]["native_phone"] = "tampered"
    jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
    # Rehash the producer receipt to model an attacker who controls hashes.
    receipt = json.loads((out / "receipt.json").read_text())
    for item in receipt.get("outputs", []):
        if Path(item.get("path", "")).name == jsonl.name:
            item["size"] = jsonl.stat().st_size
            from scripts.ja_en_schema import sha256_file
            item["sha256"] = sha256_file(jsonl)
    (out / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    report = verify_workspace(out)
    assert any(error["code"] == "tts_invalid" for error in report["errors"])


def test_textgrid_parser_unescapes_quotes_and_rejects_nan(tmp_path: Path):
    valid = tmp_path / "quoted.TextGrid"
    valid.write_text('''File type = "ooTextFile"\nObject class = "TextGrid"\nxmin = 0\nxmax = 1\ntiers? <exists>\nsize = 1\nitem []:\n    item [1]:\n        class = "IntervalTier"\n        name = "words"\n        xmin = 0\n        xmax = 1\n        intervals: size = 1\n        intervals [1]:\n            xmin = 0\n            xmax = 1\n            text = "say ""hi"""\n''', encoding="utf-8")
    parsed = _textgrid_parse(valid)
    assert parsed["words"][0]["text"] == 'say "hi"'
    invalid = tmp_path / "nan.TextGrid"
    invalid.write_text(valid.read_text(encoding="utf-8").rsplit("xmax = 1", 1)[0] + "xmax = nan\n            text = \"say \"\"hi\"\"\"\n", encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="non-finite"):
        _textgrid_parse(invalid)


def test_three_uid_bound_fixture_integrity_and_evidence_tamper(tmp_path: Path):
    """JA, EN, and one UID with both language runs use full unit evidence."""
    import copy, hashlib, zipfile
    from scripts.ja_en_schema import artifact_record, make_receipt, sha256_file
    from scripts.ja_tts_export import render_textgrid

    out = _record(tmp_path)
    base = json.loads((out / "tts_training_records.jsonl").read_text())
    rows, all_paths, alias_rows = [], [out / "supply_chain_lock.json"], []
    semantic_dir = out / "stages" / "semantic"; semantic_dir.mkdir(parents=True, exist_ok=True)
    authority = out / "authority"; authority.mkdir(parents=True, exist_ok=True)

    def make_row(uid, units):
        row = copy.deepcopy(base); row["uid"] = uid
        row["words"] = []; row["phones"] = []; row["durations"] = []
        row["locked_aliases"] = []; row["native_inventory"] = {"ja": [], "en": []}
        row["mora_graph"] = {"moras": [], "relations": []}; row["partition"] = {"expected_unit_ids": [u["unit"] for u in units], "verified": [u["unit"] for u in units], "rejected": [], "unresolved": []}
        raw_runs = []
        for index, unit in enumerate(units):
            start, end = index * 8000, (index + 1) * 8000
            alias, phone, lang = unit["alias"], unit["phone"], unit["language"]
            run_id = f"{uid}-{lang}-{index}"
            row["words"].append({"unit_id": unit["unit"], "token_id": unit["unit"], "text": unit["text"], "language": lang, "start_sample": start, "end_sample": end})
            pid = f"p-{uid}-{index}"
            row["phones"].append({"unit_id": unit["unit"], "phone_id": pid, "native_phone": phone, "phone": phone, "language": lang, "start_sample": start, "end_sample": end, "train_start_sample": start, "train_end_sample": end, "source_start_sample": start, "source_end_sample": end, "raw_interval_id": 2, "raw_interval_index": 1, "run_id": run_id, "alias": alias, "mora_ids": [f"m-{uid}-{index}"] if lang == "ja" else []})
            row["durations"].append(end - start); row["locked_aliases"].append({"alias": alias, "unit_id": unit["unit"], "language": lang, "pronunciation": [phone]})
            row["native_inventory"][lang].append(phone)
            raw = authority / f"{uid}-{index}.raw.TextGrid"
            raw.write_text(f'''File type = "ooTextFile"\nObject class = "TextGrid"\nxmin = 0\nxmax = 1\ntiers? <exists>\nsize = 2\nitem []:\n    item [1]:\n        class = "IntervalTier"\n        name = "words"\n        xmin = 0\n        xmax = 1\n        intervals: size = 1\n        intervals [1]:\n            xmin = 0\n            xmax = 1\n            text = "{alias}"\n    item [2]:\n        class = "IntervalTier"\n        name = "phones"\n        xmin = 0\n        xmax = 1\n        intervals: size = 1\n        intervals [1]:\n            xmin = 0\n            xmax = 1\n            text = "{phone}"\n''', encoding="utf-8")
            dictionary = authority / f"{uid}-{index}.dict"; dictionary.write_text(f"{alias} {phone}\n", encoding="utf-8")
            archive = authority / f"{uid}-{index}.zip"
            with zipfile.ZipFile(archive, "w") as zf: zf.writestr("metadata.json", json.dumps({"phones": [phone]}))
            raw_hash, dict_hash, archive_hash = (sha256_file(path) for path in (raw, dictionary, archive))
            raw_runs.append({"run_id": run_id, "language": lang, "unit_ids": [unit["unit"]], "raw_textgrid": {"path": str(raw), "sha256": raw_hash}, "raw_tier": "phones", "sample_rate": 16000, "crop_offset_sample": start, "ownership_start_sample": start, "ownership_end_sample": end, "locked_dict_path": str(dictionary), "locked_dict_sha256": dict_hash, "native_inventory_path": str(archive), "native_inventory_sha256": archive_hash, "aliases": [{"alias": alias, "pronunciation": [phone]}]})
            alias_rows.append({"uid": uid, "token_id": unit["unit"], "candidate_id": f"cand-{unit['unit']}", "alias": alias, "pronunciation": [phone], "language": lang, "reading": unit["text"]})
            all_paths.extend([raw, dictionary, archive])
            if lang == "ja":
                row["mora_graph"]["moras"].append({"mora_id": f"m-{uid}-{index}", "phone_ids": [pid]})
                row["mora_graph"]["relations"].append({"mora_id": f"m-{uid}-{index}", "phone_id": pid})
        row["raw_mfa"] = {"runs": raw_runs, "model_assets": {}}
        if any(u["language"] == "ja" for u in units):
            graphs = []
            for index, unit in enumerate(units):
                if unit["language"] != "ja": continue
                pid, phone = row["phones"][index]["phone_id"], unit["phone"]
                graphs.append({"uid": uid, "token_id": unit["unit"], "candidate_id": f"cand-{unit['unit']}", "semantic_phone_nodes": [{"id": f"sp-{index}", "kind": "semantic_phone", "phone": phone, "mora_ids": [f"m-{uid}-{index}"]}], "mora_nodes": [{"id": f"m-{uid}-{index}", "kind": "mora"}], "nodes": [{"id": f"sp-{index}"}, {"id": f"m-{uid}-{index}"}], "edges": [{"relation": "mora_phone", "mora_id": f"m-{uid}-{index}", "phone_id": f"sp-{index}"}], "target": {"phones": [phone]}})
            graph = semantic_dir / f"{uid}.json"; graph.write_text(json.dumps({"graphs": graphs}), encoding="utf-8")
            row["semantic_graph_path"] = str(graph); row["semantic_graph_sha256"] = sha256_file(graph); all_paths.append(graph)
        row["selected_reading"] = units[0]["text"] if len(units) == 1 else None
        row["selected_readings"] = {u["unit"]: u["text"] for u in units}
        row["reading_evidence"] = {"locks": [{"uid": uid, "token_id": u["unit"], "candidate_id": f"cand-{u['unit']}", "chosen_reading": u["text"]} for u in units]}
        lock_file = authority / f"{uid}.locked.jsonl"; lock_file.write_text("".join(json.dumps({"uid": uid, "token_id": u["unit"], "candidate_id": f"cand-{u['unit']}", "chosen_reading": u["text"], "analysis_digest": "digest", "canonical_sha256": "canonical"}) + "\n" for u in units), encoding="utf-8")
        recon_file = authority / f"{uid}.reconstruction.json"; recon_file.write_text(json.dumps({"records": [{"uid": uid, "units": [{"uid": uid, "token_id": u["unit"], "candidate_id": f"cand-{u['unit']}", "locked_reading": u["text"], "analysis_digest": "digest", "canonical_sha256": "canonical"} for u in units]}]}), encoding="utf-8")
        analysis_file = authority / f"{uid}.analysis.jsonl"; analysis_file.write_text(lock_file.read_text(encoding="utf-8"), encoding="utf-8")
        row["reading_evidence"].update({"locked_readings_path": str(lock_file), "locked_readings_sha256": sha256_file(lock_file), "reconstruction_path": str(recon_file), "reconstruction_sha256": sha256_file(recon_file), "analysis_path": str(analysis_file), "analysis_sha256": sha256_file(analysis_file)})
        all_paths.extend([lock_file, recon_file, analysis_file])
        textgrid = out / f"{uid}.TextGrid"; textgrid.write_text(render_textgrid(row), encoding="utf-8"); all_paths.append(textgrid)
        return row

    rows.extend([make_row("ja", [{"unit": "ja-0", "alias": "ja_0", "phone": "a", "language": "ja", "text": "さ"}, {"unit": "ja-1", "alias": "ja_1", "phone": "k", "language": "ja", "text": "くら"}]), make_row("en", [{"unit": "en-0", "alias": "en_0", "phone": "AH", "language": "en", "text": "the"}]), make_row("mix", [{"unit": "mix-ja", "alias": "mix_ja", "phone": "a", "language": "ja", "text": "さ"}, {"unit": "mix-en", "alias": "mix_en", "phone": "AH", "language": "en", "text": "the"}])])
    alias_path = semantic_dir / "alias_map.jsonl"; alias_path.write_text("".join(json.dumps(x) + "\n" for x in alias_rows), encoding="utf-8")
    for row in rows: row["semantic_alias_map_path"] = str(alias_path); row["semantic_alias_map_sha256"] = sha256_file(alias_path)
    all_paths.append(alias_path)
    jsonl = out / "tts_training_records.jsonl"; jsonl.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8"); all_paths.append(jsonl)
    manifest = out / "input_manifest.jsonl"; manifest.write_text("".join(json.dumps({"uid": x["uid"], "wav": rows[0]["train_wav"]["path"], "text": "fixture"}) + "\n" for x in rows), encoding="utf-8"); all_paths.append(manifest)
    license_file = out / "LICENSE.txt"; license_file.write_text("fixture\n", encoding="utf-8"); all_paths.append(license_file)
    lock = {"schema": "ja-supply-chain-lock-v1", "status": "frozen", "resources": [{"id": "fixture", "kind": "model", "source_url": "fixture://model", "revision": "v1", "license": {"status": "reviewed", "path": str(license_file), "file_sha256": sha256_file(license_file)}, "artifacts": []}]}; (out / "supply_chain_lock.json").write_text(json.dumps(lock), encoding="utf-8")
    stages = ["inventory", "audio", "asr", "reading", "frontend", "semantic", "anchors", "align", "merge", "tts", "verify"]
    for name in stages:
        stage = out / "stages" / name; stage.mkdir(parents=True, exist_ok=True)
        (stage / "receipt.json").write_text(json.dumps(make_receipt(stage=name, status="RUNNING" if name == "verify" else "COMPLETE", outputs=[alias_path] if name == "semantic" else [])), encoding="utf-8")
    root_receipt = make_receipt(stage="verify", status="PENDING", inputs={"stages": stages, "artifacts": [artifact_record(manifest), artifact_record(out / "supply_chain_lock.json")]}, outputs=all_paths)
    (out / "receipt.json").write_text(json.dumps(root_receipt), encoding="utf-8")
    good = verify_workspace(out)
    if not good["integrity_ok"]:
        print(good["errors"])
        raise AssertionError(good["errors"])
    rows[2]["phones"][0]["native_phone"] = "tampered"
    jsonl.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    bad = verify_workspace(out); assert bad["integrity_ok"] is False
