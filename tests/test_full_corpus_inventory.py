import hashlib
import re
import json
from pathlib import Path

from scripts.full_corpus_inventory import scan_sources, stable_run_stem, write_frozen_inventory


def _audio(path: Path):
    import numpy as np
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.r_[np.zeros(800), np.ones(800)], 16000)


def test_scan_sources_applies_all_layout_rules(tmp_path):
    gd = tmp_path / "gamedata"; wu = tmp_path / "wuwa"; v5 = tmp_path / "v5"
    _audio(gd / "崩铁" / "白露" / "game_ref.wav")
    (gd / "崩铁" / "白露" / "game_ref.txt").write_text("你好")
    _audio(wu / "今汐" / "wuwa_ref.wav")
    (wu / "今汐" / "wuwa_ref.lab").write_text("「你好」~")
    _audio(wu / "其它语音 - Others" / "other.wav")
    _audio(wu / "带变量语音 - Placeholder" / "placeholder.wav")
    _audio(v5 / "LAria" / "wavs" / "laria_noref.wav")
    _audio(v5 / "中文角色" / "chinese0707.wav")
    snapshot = scan_sources({"sources": {"gamedata": gd, "wuwa": wu, "v5_0707": v5}})
    by_original = {row.original_stem: row for row in snapshot.items}
    assert by_original["game_ref"].text_mode == "reference"
    assert by_original["game_ref"].reference_suffix == ".txt"
    assert by_original["wuwa_ref"].speaker == "今汐"
    assert by_original["wuwa_ref"].reference_suffix == ".lab"
    assert by_original["laria_noref"].text_mode == "fallback"
    assert {row.original_stem for row in snapshot.excluded} >= {"other", "placeholder", "chinese0707"}


def test_stable_id_is_path_stable_and_frozen_inventory_is_serializable(tmp_path):
    first = stable_run_stem("gamedata", "崩铁", "白露", "白露/a.wav")
    second = stable_run_stem("gamedata", "崩铁", "白露", "白露/a.wav")
    assert first == second and re.fullmatch(r"u[0-9a-f]{32}", first)
    root = tmp_path / "g"; _audio(root / "a.wav")
    (tmp_path / "w").mkdir(); (tmp_path / "v").mkdir()
    snap = scan_sources({"sources": {"gamedata": root, "wuwa": tmp_path / "w", "v5_0707": tmp_path / "v"}})
    payload = write_frozen_inventory(snap, tmp_path / "inventory.json")
    assert payload["items"][0]["audio_sha256"] == hashlib.sha256((root / "a.wav").read_bytes()).hexdigest()


def test_sample_only_is_bounded_per_source_and_does_not_write_sources(tmp_path):
    roots = {}
    for key, speaker in (("gamedata", "g"), ("wuwa", "今汐"), ("v5_0707", "LAria")):
        root = tmp_path / key; roots[key] = root
        for index in range(4):
            audio = (root / "game" / speaker / f"a{index}.wav" if key == "gamedata"
                     else root / speaker / f"a{index}.wav")
            _audio(audio)
    before = {key: sorted(path.rglob("*")) for key, path in roots.items()}
    snapshot = scan_sources({"sources": roots, "_sample_only": 2})
    assert {row.source_id for row in snapshot.items} == set(roots)
    assert all(sum(row.source_id == source for row in snapshot.items) <= 2 for source in roots)
    assert {key: sorted(path.rglob("*")) for key, path in roots.items()} == before


def test_frozen_inventory_is_write_once(tmp_path):
    root = tmp_path / "g"; _audio(root / "a.wav")
    (tmp_path / "w").mkdir(); (tmp_path / "v").mkdir()
    snap = scan_sources({"sources": {"gamedata": root, "wuwa": tmp_path / "w", "v5_0707": tmp_path / "v"}})
    path = tmp_path / "inventory.json"
    write_frozen_inventory(snap, path)
    changed = json.loads(path.read_text())
    changed["schema"] = "tampered"
    path.write_text(json.dumps(changed))
    try:
        write_frozen_inventory(snap, path)
    except ValueError as exc:
        assert "different bytes" in str(exc)
    else:
        raise AssertionError("tampered frozen inventory was accepted")
