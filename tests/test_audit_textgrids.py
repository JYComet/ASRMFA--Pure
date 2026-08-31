import pytest

from scripts import audit_textgrids as audit


def _intervals(labels):
    return [
        {"xmin": float(index), "xmax": float(index + 1), "text": label}
        for index, label in enumerate(labels)
    ]


def _textgrid(*, words=None, phones=None, pinyin_phones=None,
              raw_text="<sp1>你好", pinyin=None, hanzi=None):
    words = list(words or ["<sp1>", "ni3", "hao3"])
    pinyin = pinyin or "<sp1> " + " ".join(
        label for label in words if not audit.SP_TOKEN_PAT.fullmatch(label)
    )
    hanzi = list(hanzi or ["<sp1>"] + ["字"] * (len(words) - 1))
    phones = list(pinyin_phones or phones or
                  (["<sp1>"] + ["n"] * (len(words) - 1)))
    duration = float(max(len(words), len(phones), len(hanzi), 1))

    def tier(labels):
        return {"xmin": 0.0, "xmax": duration, "intervals": _intervals(labels)}

    return {
        "filename": "fixture.TextGrid",
        "xmin": 0.0,
        "xmax": duration,
        "tiers": {
            "raw_text": {"xmin": 0.0, "xmax": duration,
                         "intervals": [{"xmin": 0.0, "xmax": duration,
                                         "text": raw_text}]},
            "pinyin": {"xmin": 0.0, "xmax": duration,
                       "intervals": [{"xmin": 0.0, "xmax": duration,
                                       "text": pinyin}]},
            "hanzi": tier(hanzi),
            "words": tier(words),
            "pinyin_phones": tier(phones),
        },
    }


def _issues(monkeypatch, tg):
    monkeypatch.setattr(audit, "parse_textgrid", lambda _: tg)
    return audit.audit_file("fixture.TextGrid")


def _categories(issues):
    return [issue.category for issue in issues]


def test_sentence_initial_sp1_is_allowed_in_fine_grained_tiers(monkeypatch):
    issues = _issues(monkeypatch, _textgrid())

    assert "special_token_leak" not in _categories(issues)


@pytest.mark.parametrize(
    "tier_name, labels",
    [
        ("words", ["ni3", "<sp1>", "hao3"]),
        ("words", ["ni3", "<sp1>"]),
        ("words", ["<sp0>", "ni3"]),
        ("words", ["<sp1>", "<sp1>", "ni3"]),
        ("hanzi", ["字", "<sp1>"]),
        ("hanzi", ["<sp3>", "字"]),
        ("hanzi", ["<sp1>foo", "字"]),
        ("pinyin_phones", ["n", "<sp1>", "a"]),
        ("pinyin_phones", ["<sp2>", "n"]),
        ("pinyin_phones", ["<sp1>", "<sp1>", "n"]),
        ("pinyin_phones", ["<sp1>foo", "n"]),
    ],
)
def test_internal_tail_duplicate_and_noncanonical_sp_are_reported(
        monkeypatch, tier_name, labels):
    kwargs = {tier_name: labels}
    if tier_name == "words":
        kwargs["hanzi"] = ["<sp1>"] + ["字"] * (len(labels) - 1)
    tg = _textgrid(**kwargs)

    issues = _issues(monkeypatch, tg)

    assert "special_token_leak" in _categories(issues)


@pytest.mark.parametrize(
    "tier_name, labels",
    [
        ("hanzi", ["foo<sp1>", "字"]),
        ("words", ["<sp1>foo", "ni3"]),
        ("pinyin_phones", ["n<sp1>", "a"]),
    ],
)
def test_embedded_sp_token_in_fine_grained_interval_is_reported(
        tier_name, labels):
    tg = _textgrid(**{tier_name: labels})

    leaks = audit._special_token_leaks(tg["tiers"])

    assert any(leak.startswith(f"{tier_name}[") for leak in leaks)


@pytest.mark.parametrize(
    "tier_name, text",
    [
        ("raw_text", "你<sp1>好"),
        ("raw_text", "你好<sp1>"),
        ("raw_text", "<sp1>你好<sp1>"),
        ("pinyin", "ni3 <sp1> hao3"),
        ("pinyin", "ni3 hao3 <sp1>"),
        ("pinyin", "<sp1> ni3 <sp1> hao3"),
    ],
)
def test_summary_sp1_must_be_one_sentence_initial_prefix(
        monkeypatch, tier_name, text):
    kwargs = {tier_name: text}
    issues = _issues(monkeypatch, _textgrid(**kwargs))

    assert "special_token_leak" in _categories(issues)


def test_summary_sentence_initial_sp1_is_allowed(monkeypatch):
    tg = _textgrid(raw_text="<sp1>你好", pinyin="<sp1> ni3 hao3")

    issues = _issues(monkeypatch, tg)

    assert "special_token_leak" not in _categories(issues)


def test_tone_marked_pinyin_is_not_reported_as_english(monkeypatch):
    tg = _textgrid(words=["<sp1>", "da4", "jia1"],
                   phones=["<sp1>", "d", "j"],
                   hanzi=["<sp1>", "大", "家"],
                   pinyin="<sp1> da4 jia1")

    issues = _issues(monkeypatch, tg)

    assert "language_mixing" not in _categories(issues)


@pytest.mark.parametrize("label", ["hello", "target1"])
def test_real_english_alpha_alnum_is_still_reported(monkeypatch, label):
    tg = _textgrid(words=["<sp1>", label], phones=["<sp1>", "HH"],
                   hanzi=["<sp1>", "字"],
                   pinyin=f"<sp1> {label}")

    issues = _issues(monkeypatch, tg)

    assert "language_mixing" in _categories(issues)
