import json
from pathlib import Path

import pytest

from scripts.ja_asr_crossval import (
    ASR_PROFILES,
    ProviderResult,
    family_vote_rows,
    run_provider_command,
    validate_profile,
    provider_specs,
)


def test_profiles_cap_votes_by_family():
    assert ASR_PROFILES["baseline_3family"] == ("qwen", "whisper", "reazon")
    rows = [
        {"family": "whisper", "candidate": "さくら", "provider": "whisper-large-v3"},
        {"family": "whisper", "candidate": "さくら", "provider": "kotoba-v2"},
        {"family": "qwen", "candidate": "さくら", "provider": "qwen3-asr"},
    ]
    votes = family_vote_rows(rows)
    assert votes["さくら"] == {"qwen", "whisper"}


def test_qwen_only_is_dev_and_baseline_requires_three_families():
    assert validate_profile("qwen_only_dev").production is False
    with pytest.raises(ValueError, match="three"):
        validate_profile("baseline_3family", available_families={"qwen"})


def test_provider_subprocess_has_no_prompt_argument_and_preserves_failure(tmp_path):
    result = run_provider_command(
        ["python", "-c", "import json; print(json.dumps({'text':'桜'}))"],
        tmp_path / "a.wav",
        family="qwen",
        provider="qwen3-asr",
    )
    assert isinstance(result, ProviderResult)
    assert result.asr_text == "桜"
    failed = run_provider_command(
        ["python", "-c", "raise SystemExit(7)"],
        tmp_path / "a.wav",
        family="whisper",
        provider="whisper-large-v3",
    )
    assert failed.status == "failed"
    assert failed.error["returncode"] == 7


def test_reazon_variants_share_one_family_and_profile_needs_each_family():
    rows = [
        {"family": "reazon", "candidate": "せい", "provider": "reazonspeech-nemo-v2"},
        {"family": "reazon", "candidate": "せい", "provider": "reazonspeech-k2-v2"},
    ]
    assert family_vote_rows(rows)["せい"] == {"reazon"}
    specs = provider_specs({"asr": {"profile": "extended_5model", "family_vote_policy": "one_per_family"}})
    assert [spec["family"] for spec in specs][-2:] == ["whisper", "reazon"]


def test_provider_specs_builds_worker_commands_from_runtime_and_model():
    specs = provider_specs({"asr": {"profile": "qwen_only_dev", "family_vote_policy": "one_per_family",
                                      "qwen_model": "/models/qwen"}})
    assert specs[0]["command"]
    assert "/models/qwen" in specs[0]["command"]


def test_raw_asr_text_without_kana_candidates_cannot_vote():
    from scripts.ja_asr_crossval import family_vote_rows
    assert family_vote_rows([
        {"provider": "qwen3-asr", "family": "qwen", "status": "ok", "asr_text": "生"},
        {"provider": "whisper-large-v3", "family": "whisper", "status": "ok", "asr_text": "生"},
    ]) == {}


def test_provider_specs_use_configured_runtime_and_language():
    specs = provider_specs({"asr": {"profile": "qwen_only_dev", "family_vote_policy": "one_per_family",
                                      "qwen_model": "/models/qwen", "qwen3-asr_runtime": "/bin/python-x",
                                      "language": "English"}})
    assert specs[0]["command"][0] == "/bin/python-x"
    assert "English" in specs[0]["command"]
