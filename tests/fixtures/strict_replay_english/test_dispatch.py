#!/usr/bin/env python3
"""Direct v2.1 English-dispatch fixtures.

These fixtures deliberately call the production verifier module and inspect
the real audit import boundary; they do not replace either call with a
runner stub or consume a pipeline/pilot artifact.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import verify_strict_replay_english_subset as verifier  # noqa: E402


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_v21(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    workspace = root / "mfa-dev-workspace"
    workspace.mkdir()
    replay_path = workspace / "strict_replay_import.json"
    subset_path = workspace / "strict_replay_english_alignment_subset.json"
    parent_path = workspace / "en_phones" / "en_alignment_manifest.json"
    parent_path.parent.mkdir()
    config_path = workspace / "config.yaml"
    config_path.write_text("fixture: dispatch\n", encoding="utf-8")
    dictionary_paths = []
    roles = {}
    for name in ("chinese_mfa_dictionary", "pinyin_projection_dictionary",
                 "english_pronunciation_dictionary"):
        path = workspace / f"{name}.dict"
        path.write_text(name + "\n", encoding="utf-8")
        dictionary_paths.append(path)
        roles[name] = {"path": str(path), "sha256": sha(path)}

    parent_bytes = b'{"schema":"strict-en-mfa-v1","strict_provenance":true}\n'
    source_parent = workspace / "authoritative-parent.json"
    source_parent.write_bytes(parent_bytes)
    parent_path.write_bytes(parent_bytes)
    parent_role = lambda path: {"path": str(path), "sha256": sha(path),
                                "immutable_import_path": str(path),
                                "immutable_import_sha256": sha(path)}
    parent_global = {"authoritative_source": parent_role(source_parent),
                     "workspace_copy": parent_role(parent_path),
                     "content_identity_sha256": sha(source_parent)}
    subset_path.write_text(json.dumps({
        "schema": verifier.ALIGNMENT_SUBSET_V21_SCHEMA,
        "parent_global_manifest": {
            "authoritative_source": {"path": str(source_parent)},
            "workspace_copy": {"path": str(parent_path)},
        },
    }), encoding="utf-8")

    stems = [f"stem{i:02d}" for i in range(21)]
    excluded = stems[-3:]
    eligible = stems[:-3]
    mapping = [{"slot": i, "stem": f"stem{i:02d}"} for i in range(96)]
    selected = mapping[:24]
    replay = {
        "schema": verifier.REPLAY_V21_SCHEMA,
        "selection_slot_records": selected,
        "slot_stem_mapping": mapping,
        "assets": [{"stem": stem} for stem in stems],
        "missing_mfa_alignment": excluded,
    }
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    records = [{"stem": stem} for stem in eligible]
    payload = {
        "schema": verifier.ENGLISH_IMPORT_V21_SCHEMA, "scope": "strict_replay",
        "run_id": "fixture", "timestamp_utc": "2026-01-01T00:00:00Z",
        "canonical_manifest_path": str(workspace / "canonical.json"),
        "canonical_manifest_sha256": "0" * 64,
        "replay_import_manifest_path": str(replay_path),
        "replay_import_manifest_sha256": sha(replay_path),
        "selection_slot_records": selected, "selection_slot_count": 24,
        "selection_slot_digest": digest(selected),
        "source_stems": stems, "source_count": 21, "source_digest": digest(stems),
        "exclusion_records": [{"stem": stem, "reason": "missing_mfa_alignment"} for stem in excluded],
        "exclusion_count": 3,
        "exclusion_digest": digest([{"stem": stem, "reason": "missing_mfa_alignment"} for stem in excluded]),
        "excluded_stems": excluded, "excluded_count": 3, "excluded_digest": digest(excluded),
        "eligible_stems": eligible, "eligible_count": 18, "eligible_digest": digest(eligible),
        "english_required_stems": eligible, "english_required_count": 18, "english_required_digest": digest(eligible),
        "english_entries_stems": eligible, "english_entries_count": 18, "english_entries_digest": digest(eligible),
        "producer_revision": "fixture", "config_sha256": sha(config_path),
        "dictionary_roles": roles, "dictionary_roles_digest": digest(roles),
        "parent_global_manifest": parent_global,
        "english_alignment_subset_path": str(subset_path),
        "english_alignment_subset_sha256": sha(subset_path),
        "parent_english_manifest_path": str(parent_path),
        "parent_english_manifest_sha256": sha(parent_path),
        "records": records,
    }
    import_path = workspace / "strict_replay_english_import.json"
    import_path.write_text(json.dumps(payload), encoding="utf-8")
    return import_path, replay_path, subset_path, parent_path, dictionary_paths[-1]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="strict-replay-dispatch-") as raw:
        import_path, replay_path, subset_path, parent_path, dictionary_path = make_v21(Path(raw))
        config_path = import_path.parent / "config.yaml"
        errors = verifier.verify_english_import_active(
            import_path, replay_path=replay_path, subset_path=subset_path,
            parent_path=parent_path, subset_sha256=sha(subset_path),
            parent_sha256=sha(parent_path), config_path=config_path,
            dictionary_path=dictionary_path)
        if errors:
            print("valid v2.1 fixture rejected:", errors)
            return 1
        replay = json.loads(replay_path.read_text())
        replay["slots"] = replay.pop("slot_stem_mapping")
        replay_path.write_text(json.dumps(replay))
        errors = verifier.verify_english_import_active(import_path, replay_path=replay_path)
        if not any("slot_stem_mapping" in error for error in errors):
            print("slots-only fixture did not expose slot_stem_mapping error:", errors)
            return 1

    source = (SCRIPTS / "audit_strict_ok.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               and node.module == "verify_strict_replay_english_subset"]
    names = {alias.name for node in imports for alias in node.names}
    if "verify_english_import_active" not in names or any(name.startswith("_") for name in names):
        print("audit private/public import boundary invalid:", sorted(names))
        return 1

    env = dict(os.environ); env["PYTHONPATH"] = str(SCRIPTS)
    probe = subprocess.run([sys.executable, "-c",
                            "import verify_strict_replay_english_subset as m; print(m.__file__)"] ,
                           env=env, capture_output=True, text=True)
    if probe.returncode or Path(probe.stdout.strip()).resolve() != (SCRIPTS / "verify_strict_replay_english_subset.py").resolve():
        print("mfa-dev import resolution failed:", probe.stdout, probe.stderr)
        return 1
    print("dispatch fixtures PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
