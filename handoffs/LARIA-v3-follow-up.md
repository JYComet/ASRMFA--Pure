# LAria v3 deferred follow-up

This handoff is documentation only. It does not authorize Qwen implementation,
Qwen invocation, NVASR invocation, or any production rerun.

## Deferred work

1. Evaluate Qwen lexical and punctuation handling against the frozen LAria
   stems. Record token/word order, punctuation placement, and any changed
   lexical decisions as reviewable evidence.
2. Evaluate Qwen forced-aligner timestamps independently. Preserve input audio
   identity, timestamp units, per-stem status, and rejected-stem reasons.
3. Define and test an NVASR gap-constrained NVV comparison. The comparison
   must identify the allowed gap, candidate NVV, owner stem, and acceptance or
   rejection reason; it must not silently change the current v3 dispatch policy.
4. Require model, model-hash, audio, and fusion receipts for every A/B arm.
   Receipts must bind the exact model/config inputs, audio bytes or hashes,
   lexical/punctuation inputs, timestamp producer, NVV evidence, and fusion
   result to the same stem set.
5. Conduct an A/B review of the current v3 path versus the proposed follow-up
   evidence. Review lexical correctness, punctuation, timestamp plausibility,
   NVV gap behavior, and failure/shortfall accounting per stem.

## Acceptance shape

The follow-up is ready for a separately authorized implementation decision only
when both A/B arms have complete, hash-bound receipts and an exact per-stem
comparison. Any shortfall remains explicitly quarantined with hard evidence;
aggregate counts alone are insufficient.

## Explicit non-goals

Do not add Qwen dependencies, code paths, CLI invocations, model downloads,
audio transformations, or production artifacts as part of this follow-up.
