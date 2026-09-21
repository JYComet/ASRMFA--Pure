#!/usr/bin/env python3
"""Small subprocess boundary for local ASR runtimes.

The worker accepts only a provider, model path and WAV.  It emits one JSON
object and never accepts a text prompt, preventing a transcript from leaking
into an ostensibly blind ASR call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _qwen(model: str, wav: Path, *, device: str = "cuda:0", dtype: str = "bfloat16", language: str | None = None) -> str:
    try:
        from qwen_asr import Qwen3ASRModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError("qwen_asr is unavailable; install qwen_asr in the configured runtime") from exc
    import torch
    kwargs: dict[str, Any] = {"device_map": device}
    if dtype == "bfloat16":
        kwargs["dtype"] = torch.bfloat16
    elif dtype == "float16":
        kwargs["dtype"] = torch.float16
    loaded = Qwen3ASRModel.from_pretrained(model, **kwargs)
    result = loaded.transcribe(audio=str(wav), language=None if not language or language.lower() == "auto" else language, context="")
    if isinstance(result, str):
        return result
    if isinstance(result, list) and result:
        first = result[0]
        text = first.get("text", first.get("asr_text", "")) if isinstance(first, dict) else getattr(first, "text", "")
        if text:
            return str(text)
    raise RuntimeError("Qwen runtime returned no transcript")


def _whisper(model: str, wav: Path, *, language: str | None = None) -> str:
    if Path(model).expanduser().is_dir():
        return _kotoba(model, wav, language=language)
    try:
        import whisper  # type: ignore
    except ImportError as exc:
        raise RuntimeError("openai-whisper is unavailable in the configured runtime") from exc
    loaded = whisper.load_model(model)
    result = loaded.transcribe(str(wav), language=language or "ja", fp16=False)
    text = result.get("text") if isinstance(result, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Whisper runtime returned no transcript")
    return text.strip()


def _kotoba(model: str, wav: Path, *, language: str | None = None) -> str:
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline  # type: ignore
        import torch
    except ImportError as exc:
        raise RuntimeError("transformers is unavailable for Kotoba") from exc
    processor = AutoProcessor.from_pretrained(model)
    loaded = AutoModelForSpeechSeq2Seq.from_pretrained(model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    pipe = pipeline("automatic-speech-recognition", model=loaded, tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor, device=0 if torch.cuda.is_available() else -1)
    result = pipe(str(wav), generate_kwargs={"language": language or "ja"})
    text = result.get("text") if isinstance(result, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Kotoba runtime returned no transcript")
    return text.strip()


def _reazon(model: str, wav: Path, *, variant: str) -> str:
    model_path = Path(model).expanduser()
    if not model_path.exists() or model_path.is_symlink():
        raise RuntimeError(f"configured Reazon local model is missing: {model_path}")
    if variant == "k2":
        try:
            from reazonspeech.k2.asr import transcribe, audio_from_path  # type: ignore
        except ImportError as exc:
            raise RuntimeError("reazonspeech.k2 is unavailable in the configured runtime") from exc
        import sherpa_onnx  # type: ignore
        required = {
            "tokens": model_path / "tokens.txt",
            "encoder": model_path / "encoder-epoch-99-avg-1.onnx",
            "decoder": model_path / "decoder-epoch-99-avg-1.onnx",
            "joiner": model_path / "joiner-epoch-99-avg-1.onnx",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise RuntimeError(f"Reazon K2 local model files are missing: {missing}")
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=str(required["tokens"]), encoder=str(required["encoder"]),
            decoder=str(required["decoder"]), joiner=str(required["joiner"]),
            num_threads=1, sample_rate=16000, feature_dim=80,
            decoding_method="greedy_search", provider="cuda" if __import__("torch").cuda.is_available() else "cpu")
        loaded = recognizer
        result = transcribe(loaded, audio_from_path(str(wav)))
    else:
        try:
            from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path  # type: ignore
        except ImportError as exc:
            raise RuntimeError("reazonspeech.nemo is unavailable in the configured runtime") from exc
        if model_path.suffix == ".nemo":
            try:
                from nemo.collections.asr.models import EncDecRNNTBPEModel  # type: ignore
            except ImportError as exc:
                raise RuntimeError("NeMo is unavailable for local .nemo Reazon model") from exc
            loaded = EncDecRNNTBPEModel.restore_from(restore_path=str(model_path), map_location="cuda" if __import__("torch").cuda.is_available() else "cpu")
        else:
            # Official ReazonSpeech load_model takes device only and never
            # downloads or accepts a model_path argument.
            loaded = load_model(device="cuda" if __import__("torch").cuda.is_available() else "cpu")
        result = transcribe(loaded, audio_from_path(str(wav)))
    text = result if isinstance(result, str) else getattr(result, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("ReazonSpeech runtime returned no transcript")
    return text.strip()


def transcribe(provider: str, model: str, wav: Path, *, device: str = "cuda:0", dtype: str = "bfloat16", language: str | None = None) -> str:
    if not wav.is_file() or wav.is_symlink():
        raise RuntimeError(f"WAV is missing or symlinked: {wav}")
    if provider == "qwen3-asr":
        return _qwen(model, wav, device=device, dtype=dtype, language=language)
    if provider in {"whisper-large-v3", "kotoba-v2"}:
        return _whisper(model, wav, language=language) if provider == "whisper-large-v3" else _kotoba(model, wav)
    if provider == "reazonspeech-nemo-v2":
        return _reazon(model, wav, variant="nemo")
    if provider == "reazonspeech-k2-v2":
        return _reazon(model, wav, variant="k2")
    raise RuntimeError(f"unsupported provider: {provider}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--language", default="auto")
    args = parser.parse_args(argv)
    try:
        text = transcribe(args.provider, args.model, args.wav, device=args.device, dtype=args.dtype,
                          language=None if args.language == "auto" else args.language)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": {"reason": type(exc).__name__, "message": str(exc)}}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "ok", "text": text}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
