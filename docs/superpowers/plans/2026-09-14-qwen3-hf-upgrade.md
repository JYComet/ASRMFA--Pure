# Qwen3 HF pre-alignment upgrade

## Scope

`ctc_prealign.provider: qwen3_hf` is an opt-in producer for the existing MFA
route. It uses the native Transformers checkpoints
`Qwen3-ASR-1.7B-hf` and `Qwen3-ForcedAligner-0.6B-hf` and keeps the legacy
`nvasr` provider as the default.

The ASR model supplies text. The ForcedAligner supplies lexical timestamps:
English words and Chinese characters. It does not provide phoneme timestamps,
NVV event labels, or a replacement for MFA's phoneme alignment stage.

## Pipeline contract

The producer receives the pipeline's active `--audio-dir`, reference-text
root, output root, and frozen stem selector. Reference-authority stems skip
ASR inference and align the supplied text directly. No-reference stems run
Qwen ASR first, then align its transcript. Both paths emit the six existing
CTC artifacts used by normalization, adjustment, and MFA. NVV flags are
rejected for this provider.

The native environment is isolated in `requirements-qwen3-hf.txt`; the
legacy `qwen-asr==0.0.6` environment is not upgraded in place. The canary
configuration points at both converted `-hf` model trees and uses
`batch_size: 1`, since the pipeline currently invokes one audio item per
producer call.

## Verification boundary

The producer rejects missing model trees, non-finite or non-positive lexical
spans, and audio beyond the ForcedAligner five-minute limit. Model tree,
runtime, settings, input/reference, and output artifact identities are stored
for resume checks. Unit tests use an injected fake backend and do not download
weights or execute a production run.
