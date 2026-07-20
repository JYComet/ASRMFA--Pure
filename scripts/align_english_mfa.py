#!/usr/bin/env python3
"""
English MFA alignment — extract English segments from CTC TextGrids and align
with the english_us_arpa acoustic model (ARPABET phone set) for phoneme-level
boundaries.

Inputs:
  workspace/ctc_pretg_adj/  (or ctc_pretg)  — CTC TextGrids + .lab files
  workspace/audio_16k/                       — 16kHz mono audio
  pretrained_models/acoustic/english_us_arpa.zip
  dict/cmudict.dict
  pretrained_models/g2p/english_us_arpa.zip

Outputs:
  workspace/en_phones/{stem}_en_phones.json  — English phoneme alignments

Usage:
  python scripts/align_english_mfa.py \
      --ctc-dir workspace/ctc_pretg_adj \
      --audio-dir workspace/audio_16k \
      --output-dir workspace/en_phones \
      --acoustic-model pretrained_models/acoustic/english_us_arpa.zip \
      --dictionary dict/cmudict.dict \
      --g2p-model pretrained_models/g2p/english_us_arpa.zip \
      --temp-dir workspace/temp_en \
      --num-jobs 4
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pipeline_utils import (
    is_english_token, is_nvv_token, is_silence, SILENCE_LABELS,
    find_mfa_python, get_mfa_env,
    is_english_phone as is_arpabet_phone,
    report_en_ipa_mappings,
)

# English MFA phone inventory — vowels, consonants, and stress markers
_ENGLISH_VOWELS = {
    'AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY',
    'EH', 'ER', 'EY', 'IH', 'IX', 'IY', 'OW', 'OY', 'UH', 'UW', 'UX',
    'AA0', 'AE0', 'AH0', 'AO0', 'AW0', 'AX0', 'AY0',
    'EH0', 'ER0', 'EY0', 'IH0', 'IX0', 'IY0', 'OW0', 'OY0', 'UH0', 'UW0',
    'AA1', 'AE1', 'AH1', 'AO1', 'AW1', 'AY1',
    'EH1', 'ER1', 'EY1', 'IH1', 'IY1', 'OW1', 'OY1', 'UH1', 'UW1',
    'AA2', 'AE2', 'AH2', 'AO2', 'AW2', 'AY2',
    'EH2', 'ER2', 'EY2', 'IH2', 'IY2', 'OW2', 'OY2', 'UH2', 'UW2',
}
_ENGLISH_CONSONANTS = {
    'B', 'CH', 'D', 'DH', 'DX', 'EL', 'EM', 'EN', 'ENG', 'F', 'G',
    'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'NX', 'P', 'Q', 'R', 'S',
    'SH', 'T', 'TH', 'V', 'W', 'WH', 'Y', 'Z', 'ZH',
}
_ENGLISH_SILENCE = {'sil', 'sp', 'spn', '<eps>'}


def _read_dict(path: Path) -> str:
    """Read a pronunciation dictionary with encoding tolerance.

    Tries UTF-8 first (MFA default), falls back to latin-1 (CMUdict).
    Returns only valid dictionary lines (filters comments / blank lines).
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    # Filter out CMUdict comment/header lines — MFA chokes on them
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith((";;;", "#")):
            lines.append(stripped)
    return "\n".join(lines)


def is_english_phone(phone: str) -> bool:
    """Check if *phone* is an MFA English phone (ARPABET-based)."""
    p = phone.strip()
    return p in _ENGLISH_VOWELS or p in _ENGLISH_CONSONANTS or p in _ENGLISH_SILENCE


def parse_textgrid_simple(path: Path) -> list[dict]:
    """Parse a Praat TextGrid, return list of intervals from the first tier.

    Returns list of {xmin, xmax, text}.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    intervals = []
    in_interval = False
    pending_xmin = pending_xmax = None

    for raw in lines:
        line = raw.strip()
        if line.startswith("intervals ["):
            in_interval = True
            pending_xmin = pending_xmax = None
        elif in_interval and line.startswith("xmin = "):
            pending_xmin = float(line.split("=", 1)[1].strip())
        elif in_interval and line.startswith("xmax = "):
            pending_xmax = float(line.split("=", 1)[1].strip())
        elif in_interval and line.startswith("text = "):
            text = line.split("=", 1)[1].strip().strip('"')
            if pending_xmin is not None and pending_xmax is not None:
                intervals.append({"xmin": pending_xmin, "xmax": pending_xmax, "text": text})
            pending_xmin = pending_xmax = None
            in_interval = False

    return intervals


def find_english_segments(ctc_dir: Path, stems: list[str],
                          max_gap_s: float = 0.35) -> dict[str, list[dict]]:
    """Scan CTC TextGrids and .lab files; return English-word segments per stem.

    *max_gap_s* controls how far apart consecutive English words can be
    before they are split into separate segments (default 0.35 s).

    Returns: {stem: [{"seg_idx": 0, "words": [{"text": "hello", "start": 1.2, "end": 1.8}, ...],
                       "seg_start": 1.15, "seg_end": 1.85}]}
    """
    result: dict[str, list[dict]] = {}

    for stem in stems:
        tg_path = ctc_dir / f"{stem}.TextGrid"
        if not tg_path.exists():
            tg_path = ctc_dir / stem / f"{stem}.TextGrid"
        if not tg_path.exists():
            continue

        intervals = parse_textgrid_simple(tg_path)
        if not intervals:
            continue

        # Collect English word intervals (from words tier)
        en_words = []
        for iv in intervals:
            text = iv["text"].strip()
            if not text or text in SILENCE_LABELS or text in ("", "<eps>"):
                continue
            if is_english_token(text):
                en_words.append({"text": text, "start": iv["xmin"], "end": iv["xmax"]})

        if not en_words:
            continue

        # Merge consecutive English words into segments
        segments = []
        seg_words = [en_words[0]]
        seg_start = en_words[0]["start"]
        seg_end = en_words[0]["end"]

        for w in en_words[1:]:
            gap = w["start"] - seg_end
            # Merge if gap is small (no intervening non-English words)
            if gap < max_gap_s:
                seg_words.append(w)
                seg_end = w["end"]
            else:
                segments.append({
                    "words": seg_words,
                    "seg_start": seg_start,
                    "seg_end": seg_end,
                })
                seg_words = [w]
                seg_start = w["start"]
                seg_end = w["end"]

        if seg_words:
            segments.append({
                "words": seg_words,
                "seg_start": seg_start,
                "seg_end": seg_end,
            })

        # Assign segment indices
        for idx, seg in enumerate(segments):
            seg["seg_idx"] = idx

        result[stem] = segments

    return result


def build_en_corpus(en_segments: dict[str, list[dict]],
                    audio_dir: Path, corpus_dir: Path,
                    padding_ms: float = 50.0,
                    min_segment_dur_ms: float = 200.0) -> dict[str, list[dict]]:
    """Extract English audio segments and build MFA corpus.

    Writes {stem}_seg{idx}.wav and {stem}_seg{idx}.lab to corpus_dir.
    Returns updated en_segments with offset info.
    """
    import numpy as np

    corpus_dir.mkdir(parents=True, exist_ok=True)
    padding_s = padding_ms / 1000.0
    min_dur_s = min_segment_dur_ms / 1000.0

    for stem, segments in list(en_segments.items()):
        # Find audio file
        wav_path = audio_dir / f"{stem}.wav"
        if not wav_path.exists():
            candidates = list(audio_dir.rglob(f"{stem}.wav"))
            if candidates:
                wav_path = candidates[0]
            else:
                del en_segments[stem]
                continue

        # Read audio (scipy handles PCM float and int, mono and multi-channel)
        try:
            from scipy.io import wavfile as _wavfile
            sr, audio = _wavfile.read(str(wav_path))
        except Exception:
            del en_segments[stem]
            continue

        import numpy as np
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = audio.astype(np.float32) / 128.0 - 1.0
        else:
            audio = audio.astype(np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        total_dur = len(audio) / sr
        valid_segments = []

        for seg in segments:
            seg_start_raw = seg["seg_start"]
            seg_end_raw = seg["seg_end"]
            seg_dur = seg_end_raw - seg_start_raw

            # Skip segments that are too short for MFA
            if seg_dur < min_dur_s:
                # Still record it — postprocessing will use G2P equal split
                seg["skipped"] = True
                seg["offset"] = seg_start_raw
                valid_segments.append(seg)
                continue

            # Add padding
            seg_start_padded = max(0.0, seg_start_raw - padding_s)
            seg_end_padded = min(total_dur, seg_end_raw + padding_s)

            start_sample = int(seg_start_padded * sr)
            end_sample = int(seg_end_padded * sr)

            if end_sample <= start_sample + int(0.05 * sr):
                seg["skipped"] = True
                seg["offset"] = seg_start_raw
                valid_segments.append(seg)
                continue

            seg_audio = audio[start_sample:end_sample]
            seg_audio_int16 = (seg_audio * 32767).clip(-32768, 32767).astype(np.int16)

            seg_name = f"{stem}_seg{seg['seg_idx']}"
            seg_wav = corpus_dir / f"{seg_name}.wav"
            seg_lab = corpus_dir / f"{seg_name}.lab"

            from scipy.io import wavfile as _wavfile
            _wavfile.write(str(seg_wav), sr, seg_audio_int16)

            # .lab: English word sequence
            lab_text = " ".join(w["text"] for w in seg["words"])
            seg_lab.write_text(lab_text + "\n", encoding="utf-8")

            seg["skipped"] = False
            seg["offset"] = seg_start_padded  # padded segment start in original timeline
            seg["seg_name"] = seg_name
            valid_segments.append(seg)

        if valid_segments:
            en_segments[stem] = valid_segments
        else:
            del en_segments[stem]

    return en_segments


def build_en_dict(en_segments: dict[str, list[dict]],
                  base_dict: Path, g2p_model: Path,
                  mfa_python: Path, models_dir: Path,
                  temp_dir: Path) -> Path:
    """Build English pronunciation dictionary for the corpus.

    Checks all English words against base_dict; runs G2P for OOV words.
    Always returns a clean dictionary (comments stripped) — never the
    raw base_dict path because MFA can't parse CMUdict comment lines.
    """
    # Collect all unique English words
    all_words: set[str] = set()
    for stem, segments in en_segments.items():
        for seg in segments:
            for w in seg["words"]:
                word = w["text"].strip().lower()
                if word and word.isalpha():
                    all_words.add(word)

    if not all_words:
        return base_dict

    # Load base dictionary entries
    base_words: set[str] = set()
    base_dict_text = ""
    if base_dict.exists():
        base_dict_text = _read_dict(base_dict)
        for line in base_dict_text.splitlines():
            line = line.strip()
            if line:
                parts = line.split(None, 1)
                if parts:
                    word = parts[0].split("(")[0].lower()
                    base_words.add(word)

    oov_words = sorted(all_words - base_words)
    # Always start with a clean (comment-free) copy of the base dictionary
    combined = temp_dir / "en_combined.dict"
    if not oov_words:
        with open(combined, 'w', encoding='utf-8') as outf:
            outf.write(base_dict_text)
        return combined

    # Check dictionary cache (keyed by hash of sorted OOV word list)
    import hashlib
    cache_key = hashlib.sha1(",".join(oov_words).encode()).hexdigest()[:12]
    cache_dir = temp_dir / "en_dict_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_dict = cache_dir / f"{cache_key}.dict"
    if cached_dict.exists():
        # Merge base + cached
        combined = temp_dir / "en_combined.dict"
        with open(combined, 'w', encoding='utf-8') as outf:
            if base_dict.exists():
                outf.write(_read_dict(base_dict))
                outf.write("\n")
            outf.write(cached_dict.read_text(encoding="utf-8"))
        return combined

    # Run G2P for OOV words
    oov_file = temp_dir / "en_oov_words.txt"
    oov_file.write_text("\n".join(oov_words) + "\n", encoding="utf-8")
    g2p_output = temp_dir / "en_oov_dict.txt"

    g2p_model_path = str(g2p_model)
    if not Path(g2p_model_path).exists():
        # Try zip extension
        g2p_zip = Path(str(g2p_model) + ".zip")
        if g2p_zip.exists():
            g2p_model_path = str(g2p_zip)
        else:
            print(f"  WARNING: G2P model not found at {g2p_model}, skipping OOV generation")
            return combined

    print(f"  Running G2P for {len(oov_words)} OOV English words...")
    try:
        rc = subprocess.run(
            [str(mfa_python), "-m", "montreal_forced_aligner.command_line.mfa",
             "g2p", str(oov_file), g2p_model_path, str(g2p_output),
             "--num_pronunciations", "1", "--clean"],
            env=get_mfa_env(mfa_python, models_dir),
            timeout=300, capture_output=True, text=True,
        )
        if rc.returncode != 0:
            print(f"  WARNING: G2P failed: {rc.stderr[-500:] if rc.stderr else 'unknown'}")
            return combined
    except subprocess.TimeoutExpired:
        print("  WARNING: G2P timed out")
        return combined
    except Exception as e:
        print(f"  WARNING: G2P error: {e}")
        return combined

    if not g2p_output.exists():
        return combined

    # Merge base dict + G2P output
    combined = temp_dir / "en_combined.dict"
    with open(combined, 'w', encoding='utf-8') as outf:
        if base_dict.exists():
            outf.write(_read_dict(base_dict))
            outf.write("\n")
        outf.write(g2p_output.read_text(encoding="utf-8"))

    # Save to cache for future runs
    import shutil
    shutil.copy(g2p_output, cached_dict)

    print(f"  Combined dictionary: {combined} ({len(all_words)} words)")
    return combined


def run_en_mfa(corpus_dir: Path, dict_path: Path, acoustic_model: str,
               output_dir: Path, temp_dir: Path, mfa_python: Path,
               models_dir: Path, num_jobs: int = 4,
               beam: int = 10, retry_beam: int = 40) -> bool:
    """Run MFA align with English models. Returns True on success."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Use extracted model if available
    extracted = models_dir / "extracted_models" / "acoustic" / "english_us_arpa_acoustic"
    if extracted.is_dir():
        acoustic_arg = str(extracted)
    else:
        acoustic_arg = str(acoustic_model)

    mfa_args = [
        "align", str(corpus_dir), str(dict_path),
        acoustic_arg, str(output_dir),
        "--temporary_directory", str(temp_dir),
        "--output_format", "long_textgrid",
        "--num_jobs", str(num_jobs),
        "--single_speaker",
        "--no_tokenization",
        "--beam", str(beam),
        "--retry_beam", str(retry_beam),
        "--clean",
        "--overwrite",
    ]

    print(f"  Running English MFA align ({len(list(corpus_dir.glob('*.wav')))} segments)...")
    try:
        rc = subprocess.run(
            [str(mfa_python), "-m", "montreal_forced_aligner.command_line.mfa"] + mfa_args,
            env=get_mfa_env(mfa_python, models_dir),
            timeout=1800,
        )
        if rc.returncode != 0:
            print(f"  WARNING: English MFA returned code {rc.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print("  WARNING: English MFA timed out")
        return False
    except Exception as e:
        print(f"  WARNING: English MFA error: {e}")
        return False

    return True


def parse_en_textgrid(tg_path: Path) -> dict:
    """Parse English MFA TextGrid into a simple structure.

    Returns {words: [{text, start, end, phones: [{phone, start, end}]}]}
    Assumes tier 0 = words, tier 1 = phones.
    """
    lines = tg_path.read_text(encoding="utf-8").splitlines()
    tiers_data: list[list[dict]] = []  # each tier = list of {xmin, xmax, text}
    current_tier: list[dict] = []
    in_interval = False
    pending_xmin = pending_xmax = None
    in_items = False

    for raw in lines:
        line = raw.strip()
        if line == "item []:":
            in_items = True
        elif in_items and line.startswith("item ["):
            if current_tier:
                tiers_data.append(current_tier)
                current_tier = []
            in_interval = False
        elif in_items and line.startswith("intervals ["):
            in_interval = True
            pending_xmin = pending_xmax = None
        elif in_interval and line.startswith("xmin = "):
            pending_xmin = float(line.split("=", 1)[1].strip())
        elif in_interval and line.startswith("xmax = "):
            pending_xmax = float(line.split("=", 1)[1].strip())
        elif in_interval and line.startswith("text = "):
            text = line.split("=", 1)[1].strip().strip('"')
            if pending_xmin is not None and pending_xmax is not None:
                current_tier.append({"xmin": pending_xmin, "xmax": pending_xmax, "text": text})
            pending_xmin = pending_xmax = None
            in_interval = False

    if current_tier:
        tiers_data.append(current_tier)

    if len(tiers_data) < 2:
        return {"words": []}

    words_tier = tiers_data[0]
    phones_tier = tiers_data[1]

    # Build word list with nested phones
    words = []
    phone_idx = 0
    for w in words_tier:
        text = w["text"].strip()
        if not text or text in SILENCE_LABELS or text == "<eps>":
            continue
        w_start = w["xmin"]
        w_end = w["xmax"]

        # Collect phones within this word interval
        word_phones = []
        while phone_idx < len(phones_tier):
            p = phones_tier[phone_idx]
            if p["xmin"] >= w_end - 0.001:
                break
            if p["xmax"] > w_start + 0.001:
                p_text = p["text"].strip()
                if p_text and p_text not in ("sil", "sp", "spn", "<eps>"):
                    word_phones.append({
                        "phone": p_text,
                        "start": max(p["xmin"], w_start),
                        "end": min(p["xmax"], w_end),
                    })
            phone_idx += 1

        words.append({
            "text": text,
            "start": w_start,
            "end": w_end,
            "phones": word_phones,
        })

    return {"words": words}


# Track ARPABET phones from MFA output that fail validation.
# These are reported at the end and indicate model/dictionary mismatches.
_unknown_arpabet_phones: set[str] = set()


def collect_en_phones(en_segments: dict[str, list[dict]],
                      en_aligned_dir: Path,
                      output_dir: Path) -> int:
    """Parse English MFA TextGrids and write per-stem JSON files.

    Returns number of stems processed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_processed = 0

    for stem, segments in en_segments.items():
        en_data = []

        for seg in segments:
            seg_idx = seg["seg_idx"]

            if seg.get("skipped"):
                # G2P fallback: equal-duration split for CTC word interval
                for w in seg["words"]:
                    en_data.append({
                        "seg_idx": seg_idx,
                        "offset": 0.0,
                        "word_text": w["text"],
                        "word_start": w["start"],
                        "word_end": w["end"],
                        "phones": [],  # empty -> postprocessing uses equal split
                    })
                continue

            seg_name = seg.get("seg_name", f"{stem}_seg{seg_idx}")
            tg_path = en_aligned_dir / f"{seg_name}.TextGrid"
            if not tg_path.exists():
                # Try nested
                nested = en_aligned_dir / seg_name / f"{seg_name}.TextGrid"
                if nested.exists():
                    tg_path = nested
                else:
                    # MFA didn't produce output — fall back
                    for w in seg["words"]:
                        en_data.append({
                            "seg_idx": seg_idx,
                            "offset": 0.0,
                            "word_text": w["text"],
                            "word_start": w["start"],
                            "word_end": w["end"],
                            "phones": [],
                        })
                    continue

            parsed = parse_en_textgrid(tg_path)
            offset = seg["offset"]

            # Match English MFA words to CTC words by text and sequence position
            mfa_words = parsed.get("words", [])
            ctc_words = seg["words"]

            # Simple positional matching: MFA words should align 1:1 with CTC words
            # (same text, same order)
            mfa_idx = 0
            for ctc_w in ctc_words:
                ctc_text_lower = ctc_w["text"].strip().lower()
                # Find matching MFA word
                matched = None
                while mfa_idx < len(mfa_words):
                    mw = mfa_words[mfa_idx]
                    mfa_idx += 1
                    if mw["text"].strip().lower().rstrip('012') == ctc_text_lower.rstrip('012'):
                        matched = mw
                        break
                    # If MFA word doesn't match, check next (may have been merged/skipped)

                if matched:
                    # Map MFA phone times (relative to segment) to absolute times
                    phones_abs = []
                    for p in matched["phones"]:
                        ph = p["phone"].strip()
                        # Validate against known ARPABET phone set
                        if ph and not is_arpabet_phone(ph):
                            _unknown_arpabet_phones.add(ph)
                        phones_abs.append({
                            "phone": ph,
                            "start": round(offset + p["start"], 4),
                            "end": round(offset + p["end"], 4),
                        })

                    en_data.append({
                        "seg_idx": seg_idx,
                        "offset": round(offset, 4),
                        "word_text": ctc_w["text"],
                        "word_start": ctc_w["start"],  # CTC word boundary (original time)
                        "word_end": ctc_w["end"],
                        "en_word_start": round(offset + matched["start"], 4),  # English MFA word boundary
                        "en_word_end": round(offset + matched["end"], 4),
                        "phones": phones_abs,
                    })
                else:
                    # No match found — fall back
                    en_data.append({
                        "seg_idx": seg_idx,
                        "offset": round(offset, 4),
                        "word_text": ctc_w["text"],
                        "word_start": ctc_w["start"],
                        "word_end": ctc_w["end"],
                        "phones": [],
                    })

        if en_data:
            out_path = output_dir / f"{stem}_en_phones.json"
            out_path.write_text(
                json.dumps(en_data, ensure_ascii=False, indent=2),
                encoding="utf-8")
            n_processed += 1

    # Report diagnostics
    n_mappings = report_en_ipa_mappings()
    if _unknown_arpabet_phones:
        print(f"  WARNING: Unknown ARPABET phones from MFA output "
              f"({len(_unknown_arpabet_phones)}): "
              f"{', '.join(sorted(_unknown_arpabet_phones)[:30])}"
              f"{'…' if len(_unknown_arpabet_phones) > 30 else ''}")
    elif n_mappings == 0:
        print(f"  All English phones validated (ARPABET native, no IPA mapping triggered)")

    return n_processed


def main():
    parser = argparse.ArgumentParser(description="English MFA alignment for English segments")
    parser.add_argument("--ctc-dir", type=Path, required=True,
                        help="CTC prealignment directory (with .TextGrid + .lab files)")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="16kHz mono audio directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for English phone JSON files")
    parser.add_argument("--acoustic-model", type=str, required=True,
                        help="Path to english_us_arpa acoustic model (.zip or extracted dir)")
    parser.add_argument("--dictionary", type=str, required=True,
                        help="Path to pronunciation dictionary (e.g. dict/cmudict.dict)")
    parser.add_argument("--g2p-model", type=str, default="",
                        help="Path to english_us_arpa G2P model (.zip)")
    parser.add_argument("--temp-dir", type=Path, default=None,
                        help="Temporary directory for MFA working files")
    parser.add_argument("--num-jobs", type=int, default=4,
                        help="Number of parallel MFA jobs")
    parser.add_argument("--padding-ms", type=float, default=50.0,
                        help="Padding around English segments (ms)")
    parser.add_argument("--min-segment-dur-ms", type=float, default=200.0,
                        help="Minimum segment duration for MFA (ms)")
    parser.add_argument("--max-gap-merge-s", type=float, default=0.35,
                        help="Max gap between consecutive English words to merge (s)")
    parser.add_argument("--beam", type=int, default=10,
                        help="MFA Viterbi beam width for English alignment")
    parser.add_argument("--retry-beam", type=int, default=40,
                        help="MFA retry beam width for English alignment")
    parser.add_argument("--python", type=str, default=None,
                        help="Python interpreter with MFA installed")
    args = parser.parse_args()

    # Resolve paths
    ctc_dir = args.ctc_dir
    audio_dir = args.audio_dir
    output_dir = args.output_dir
    temp_dir = args.temp_dir or (output_dir / "temp_en_mfa")
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find MFA Python
    if args.python:
        mfa_python = Path(args.python)
    else:
        mfa_python = find_mfa_python("")
    if not mfa_python or not mfa_python.exists():
        print("ERROR: Cannot find Python with MFA installed.")
        return 1

    models_dir = PROJECT_ROOT / "models" / "mfa"

    # Discover stems from CTC directory
    stems = []
    for f in sorted(ctc_dir.glob("*.lab")):
        stems.append(f.stem)
    if not stems:
        # Try nested
        for d in sorted(ctc_dir.iterdir()):
            if d.is_dir():
                lab = d / f"{d.name}.lab"
                if lab.exists():
                    stems.append(d.name)
    if not stems:
        print(f"No .lab files found in {ctc_dir}")
        return 0

    print(f"Found {len(stems)} stems with CTC output")

    # Step 1: Find English segments
    print("Scanning for English word segments...")
    en_segments = find_english_segments(ctc_dir, stems, max_gap_s=args.max_gap_merge_s)
    n_with_en = len(en_segments)
    print(f"  {n_with_en} stems contain English words")

    if n_with_en == 0:
        print("No English words found — nothing to do.")
        return 0

    total_en_words = sum(
        sum(len(seg["words"]) for seg in segs)
        for segs in en_segments.values()
    )
    print(f"  {total_en_words} total English words")

    # Step 2: Build English corpus
    en_corpus_dir = temp_dir / "en_corpus"
    print("Extracting English audio segments...")
    en_segments = build_en_corpus(
        en_segments, audio_dir, en_corpus_dir,
        padding_ms=args.padding_ms,
        min_segment_dur_ms=args.min_segment_dur_ms,
    )

    n_segments = sum(
        sum(1 for seg in segs if not seg.get("skipped"))
        for segs in en_segments.values()
    )
    n_skipped = sum(
        sum(1 for seg in segs if seg.get("skipped"))
        for segs in en_segments.values()
    )
    print(f"  {n_segments} English segments for MFA, {n_skipped} skipped (too short)")

    if n_segments == 0:
        # Still produce output for G2P fallback
        n_done = collect_en_phones(en_segments, temp_dir / "en_aligned", output_dir)
        print(f"  Wrote fallback phone data for {n_done} stems (no MFA segments)")
        return 0

    # Step 3: Build dictionary
    print("Building English dictionary...")
    dict_path = build_en_dict(
        en_segments,
        Path(args.dictionary),
        Path(args.g2p_model) if args.g2p_model else Path(""),
        mfa_python, models_dir, temp_dir,
    )

    # Step 4: Run English MFA
    en_aligned_dir = temp_dir / "en_aligned"
    success = run_en_mfa(
        en_corpus_dir, dict_path, args.acoustic_model,
        en_aligned_dir, temp_dir / "en_mfa_work",
        mfa_python, models_dir, args.num_jobs,
        beam=args.beam, retry_beam=args.retry_beam,
    )

    if not success:
        print("  WARNING: English MFA had issues — will use fallback for affected segments")

    # Step 5: Collect results
    n_done = collect_en_phones(en_segments, en_aligned_dir, output_dir)
    print(f"  Wrote English phone data for {n_done} stems")

    # Cleanup temp corpus (keep JSON output)
    if en_corpus_dir.exists():
        shutil.rmtree(en_corpus_dir, ignore_errors=True)
    en_work = temp_dir / "en_mfa_work"
    if en_work.exists():
        shutil.rmtree(en_work, ignore_errors=True)

    print(f"Done: English MFA alignment complete ({n_done} stems)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
