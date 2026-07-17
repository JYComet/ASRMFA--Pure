#!/usr/bin/env python3
"""Generate MFA test CTC data from existing TextGrid + WAV files.

Reads source TextGrids (5-tier output) and raw WAVs, resamples audio to
16 kHz mono int16, and writes CTC-format files (.lab, .TextGrid,
_tokens.jsonl, _punct.json, _text_cn.txt) that the pipeline can consume
in ``ctc_ready`` mode.

Usage:
  python scripts/setup_test_data.py --src E:/Audio/AudioEventDetection/output/test \
      --dst output/test_en_mfa --stems 合成ria_35019 Xuehusang_00029
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy import signal


def main():
    p = argparse.ArgumentParser(description="Generate CTC test data from existing TextGrids")
    p.add_argument("--src", type=Path, required=True, help="Source directory with .wav + .TextGrid")
    p.add_argument("--dst", type=Path, required=True, help="Destination workspace root")
    p.add_argument("--stems", nargs="+", required=True, help="Stems to process")
    args = p.parse_args()

    # Lazy import pipeline utils (needs repo scripts on path)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from postprocess_textgrids import parse_textgrid
    from pipeline_utils import NVV_TO_MFA, is_english_token

    for stem in args.stems:
        src_wav = args.src / f"{stem}.wav"
        src_tg  = args.src / f"{stem}.TextGrid"
        if not src_wav.exists():
            print(f"[SKIP] {stem}: missing wav"); continue
        if not src_tg.exists():
            print(f"[SKIP] {stem}: missing TextGrid"); continue

        # ── Resample audio to 16 kHz mono int16 ──
        sr, audio = wavfile.read(str(src_wav))
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0
        mx = np.abs(audio).max()
        if mx > 0:
            audio = audio / mx
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        n16 = int(len(audio) * 16000 / sr)
        a16 = signal.resample(audio, n16).astype(np.float32)
        a16 = a16 / (np.abs(a16).max() or 1.0) * 0.95
        a16i = (a16 * 32767).clip(-32768, 32767).astype(np.int16)
        dur_s = len(a16i) / 16000
        audio_dir = args.dst / "audio_16k"
        audio_dir.mkdir(parents=True, exist_ok=True)
        wavfile.write(str(audio_dir / f"{stem}.wav"), 16000, a16i)

        # ── Parse source TextGrid ──
        tg = parse_textgrid(src_tg)
        words_ivs = raw_text = ""
        for t in tg.tiers:
            if t.name == "words":
                words_ivs = t.intervals
            elif t.name == "raw_text" and t.intervals:
                raw_text = t.intervals[0].text

        # ── Build CTC tokens + .lab ──
        tokens = []       # for _tokens.jsonl
        lab_mfa = []      # for .lab  (NVV normalised)
        for iv in words_ivs:
            text = iv.text.strip()
            if not text or text in ("<sp0>", "<sp1>", "<sp2>", "<sp3>", "sil", "<eps>", ""):
                continue
            tokens.append({"word": text,
                           "start_s": round(iv.xmin, 3),
                           "end_s":   round(iv.xmax, 3)})
            stripped = text.strip("<>")
            lab_mfa.append(NVV_TO_MFA.get(stripped, stripped)
                           if stripped.upper() in NVV_TO_MFA else text)

        # ── Extract punctuation (from words tier) ──
        punct_entries = []
        all_tokens = [{"text": iv.text.strip(), "start": iv.xmin, "end": iv.xmax}
                      for iv in words_ivs
                      if iv.text.strip() and iv.text.strip() not in
                      ("<sp0>", "<sp1>", "<sp2>", "<sp3>", "sil", "<eps>", "")]
        for i, tok in enumerate(all_tokens):
            if tok["text"] in "，。！？…、；：":
                end_s = (all_tokens[i + 1]["start"]
                         if i + 1 < len(all_tokens) else tok["end"])
                punct_entries.append({
                    "word": tok["text"],
                    "start_s": round(tok["start"], 3),
                    "end_s":   round(end_s, 3),
                })

        # ── Chinese reference text ──
        cn_text = raw_text.replace("<sp1>", "").strip()

        # ── Write CTC files to both dirs ──
        for d in [args.dst / "ctc_pretg", args.dst / "ctc_pretg_adj"]:
            d.mkdir(parents=True, exist_ok=True)

            (d / f"{stem}.lab").write_text(" ".join(lab_mfa) + "\n",
                                           encoding="utf-8")

            with open(d / f"{stem}_tokens.jsonl", "w", encoding="utf-8") as f:
                for t in tokens:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")

            (d / f"{stem}_punct.json").write_text(
                json.dumps(punct_entries, ensure_ascii=False), encoding="utf-8")

            (d / f"{stem}_text_cn.txt").write_text(cn_text + "\n",
                                                    encoding="utf-8")

            # ── Simple 2-tier CTC TextGrid ──
            lines = ["File type = \"ooTextFile\"",
                     "Object class = \"TextGrid\"", "",
                     f"xmin = 0.0", f"xmax = {dur_s:.6f}",
                     "tiers? <exists>", "size = 2",
                     "item []:",
                     "    class = \"IntervalTier\"",
                     "    name = \"words\"",
                     f"    xmin = 0.0", f"    xmax = {dur_s:.6f}",
                     f"    intervals: size = {len(words_ivs)}"]
            for i, iv in enumerate(words_ivs):
                t = iv.text.strip().strip("<>")
                if t.upper() in NVV_TO_MFA:
                    t = NVV_TO_MFA[t]
                lines += [f"    intervals [{i}]:",
                          f"        xmin = {iv.xmin:.6f}",
                          f"        xmax = {iv.xmax:.6f}",
                          f"        text = \"{t}\""]
            lines += ["    class = \"IntervalTier\"",
                      "    name = \"pauses\"",
                      f"    xmin = 0.0", f"    xmax = {dur_s:.6f}",
                      "    intervals: size = 1",
                      "    intervals [0]:",
                      f"        xmin = 0.0",
                      f"        xmax = {dur_s:.6f}",
                      "        text = \"\""]
            (d / f"{stem}.TextGrid").write_text(
                "\n".join(lines), encoding="utf-8")

        en = [t for t in lab_mfa if is_english_token(t)]
        print(f"[OK] {stem}: {dur_s:.1f}s, {len(tokens)} tokens, "
              f"{len(punct_entries)} punct, EN={en}")


if __name__ == "__main__":
    main()
