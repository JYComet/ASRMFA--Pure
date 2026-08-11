from __future__ import annotations

from pathlib import Path

from scripts import run_pipeline


def _fixture(tmp_path: Path, count: int = 201):
    source = tmp_path / "source"
    shard = tmp_path / "shard"
    source.mkdir(); shard.mkdir()
    stems = [f"stem_{index:03d}" for index in range(count)]
    tasks = []
    rows = []
    for index, stem in enumerate(stems):
        src = source / f"{stem}.wav"
        dst = shard / f"{stem}.wav"
        src.write_bytes(bytes([index % 251]))
        dst.symlink_to(src)
        tasks.append((src, dst))
        rows.append({"stem": stem, "path": str(src.resolve()),
                     "sha256": run_pipeline._sha256_file(src)})
    return stems, tasks, {"audio": rows}, source


def test_large_shard_axis_links_and_hash_tamper(tmp_path):
    stems, tasks, receipt, source = _fixture(tmp_path)
    assert run_pipeline._validate_mfa_shard_axis_links(stems, tasks, receipt) == []

    (source / "stem_100.wav").write_bytes(b"tampered")
    errors = run_pipeline._validate_mfa_shard_axis_links(stems, tasks, receipt)
    assert any("hash mismatch" in error for error in errors)


def test_shard_axis_link_missing_or_wrong_target_fails(tmp_path):
    stems, tasks, receipt, source = _fixture(tmp_path, count=201)
    tasks[0][1].unlink()
    errors = run_pipeline._validate_mfa_shard_axis_links(stems, tasks, receipt)
    assert any(token in error for error in errors
               for token in ("binding missing", "target mismatch", "unreadable"))
