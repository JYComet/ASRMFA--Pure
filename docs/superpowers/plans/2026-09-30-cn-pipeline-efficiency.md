# Chinese Pipeline Efficiency Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. Steps use checkbox syntax for tracking.

**Goal:** Implement approved option A without changing alignment or publication content.

**Architecture:** Share file inventories only within one CTC receipt commit and one publication transaction. Retain independent validation at consumer, resume and final publication boundaries. Propagate explicit CPU budgets to stage pools while retaining acoustic parameters and corpus partitions.

**Tech Stack:** Python, pathlib, subprocess, concurrent.futures, pytest.

**Spec:** `docs/PIPELINE_EFFICIENCY_ANALYSIS_20260930.md`, option A, approved by the user's instruction “实现优化”.

## Global Constraints

- Preserve all existing uncommitted JA and Qwen changes; work on the current non-main feature branch with disjoint ownership.
- Keep stage ordering, CLI single-step/resume semantics, raw/work physical separation and all independent audits.
- Keep model, dictionary, beam, dither, fine-tune, corpus partition and retry algorithms unchanged.
- No cross-run hash cache, schema migration, production configuration changes or production model execution.
- Execution is authorized. Complete the implementation and verification without another approval round.

## Review Focus

- Mutation between artifact inventory and receipt commit must not be silently certified (task 1).
- Optional reference files and extra work metadata must remain in the same digest projections (task 1).
- Previously copied siblings may be corrupted before final publication or during resume (task 2).
- Symlinks and conflicting existing output files must remain rejected (task 2).
- Explicit oversized worker settings, tiny CPU budgets and nested subprocesses must obey the effective budget (task 3).

## Task 1 CTC commit inventory

**Files:** `scripts/pipeline_utils.py` CTC receipt helpers; `scripts/run_pipeline.py` CTC commit/freshness helpers; `tests/test_ctc_commit_efficiency.py`.

**Interfaces:** A commit-local `CtcWorkCommitSnapshot` binds root, raw identity and file size/hash records. `write_ctc_work_receipt` accepts an optional snapshot; existing callers keep independent validation. `_record_ctc_work_stage` derives ledger, files/ref and freshness outputs from that same inventory.

- [x] Add hash-count tests on real temporary files for 1/10/100 stems; each work artifact is hashed once per commit.
- [x] Add old-format receipt/ledger/freshness oracle comparisons and mutation, missing-file and symlink rejection tests; run red.
- [x] Implement the commit-local inventory without altering normalization functions or their order.
- [x] Run new tests and CTC lifecycle/checkpoint tests; root reviews the patch.

## Task 2 Publication manifests

**Files:** `scripts/streaming_pipeline.py` publication helpers only; publication copy/contract tests.

**Interfaces:** Copy helpers return the validated manifest for the current source batch. Staging assembles evidence from those manifests. Final dataset verification compares actual terminal output/filtered file contents with the union of trusted batch evidence.

- [x] Add the 10 one-file batch probe as a regression test: copying a new batch does not rehash old siblings.
- [x] Add target corruption, final/resume corruption, symlink, conflict and incomplete-copy tests; run red.
- [x] Reuse manifests at the same publication point; retain independent post-rename verification and full resume validation.
- [x] Verify final dataset target hashes, not merely stem membership, before accepting COMPLETE.
- [x] Run publication and orchestration tests; root reviews the patch.

## Task 3 Stage CPU budgets

**Files:** new `scripts/resource_budget.py`; streaming scheduling/commands; run_pipeline stage arguments; adjust, English alignment preparation, normalization and postprocess worker selection; resource tests.

**Dependencies:** The new resource helper and its tests can proceed independently. Tasks 1 and 2 release their shared-file responsibility regions before overlapping edits; root may release disjoint parser/worker-argument regions earlier.

**Interfaces:** One worker-budget resolver clamps explicit and automatic requests by caller budget, item count and existing stage cap. A positive budget is passed explicitly from the outer scheduler into each pipeline subprocess and onward to CPU stage pools. Existing standalone defaults apply when no outer budget is supplied.

`WeightedAdmissionGate(capacity).lease(weight, cancel_event)` provides FIFO admission in pipelined mode; cancellation raises `PhaseAdmissionCancelled`. CPU phase weight covers the maximum of unchanged MFA jobs, English jobs and auxiliary cap; GPU phase weight is its explicit auxiliary cap (default 1). Acquire after dequeue, release before enqueue, and hold no other locks while awaiting admission. Short condition waits observe cancellation; exceptions release the lease. A capacity of one serializes phases without changing their algorithms. Ordinary streaming uses its per-worker budget without this gate.

- [x] Test budget matrices, invalid input and CLI propagation into real command builders; run red.
- [x] Bound all stage pools, including producer-side normalization, without changing transformation algorithms.
- [x] Cover tiny-budget scheduling and failure wakeup so admission limits cannot deadlock bounded queues.
- [x] Run resource, launcher, MFA partition and postprocess integration tests.

## Integration and validation

- [x] Preserve original scripts snapshot: `/tmp/mfa-efficiency-baseline-cv4xdbs6/scripts`.
- [x] Full baseline: 1723 passed, 37 skipped, 1 existing out-of-scope JA verifier fixture failure, 37.28 s. Log: `/tmp/mfa-efficiency-baseline-pytest.log`.
- [x] Run the three task test groups and full suite; identify any newly introduced failure separately from the known JA fixture failure.
- [x] Run `tests/benchmark_pipeline_efficiency.py` for three alternating baseline/candidate repetitions. Compare work/publication bytes and receipt fields; whitelist only changed implementation identity fields.
- [x] Perform final correctness review and address findings before reporting completion.
- [x] Update the analysis document with measured operation counts, local microbenchmark results and limits. No production speedup claim.

## Execution ledger

- User approved option A implementation; root uses the existing feature branch to preserve current uncommitted context.
- Two independent workers implement tasks 1 and 2; task 3 follows to avoid shared-file races.
- Ruling: the current final dataset loader checks membership but does not compare terminal TextGrid bytes to staging evidence. Task 2 must add this comparison before it can safely reduce intermediate scans; no receipt schema change is needed.
- Ruling: retain existing MFA jobs and corpus partitions; use weighted phase admission to account for simultaneous producer CPU work rather than silently changing MFA parallelism. The tradeoff is possible waiting behind a large FIFO request, covered by cancellation and small-budget tests.


## Final evidence

- CTC legacy receipt oracle, 1/10/100-stem hash counts, mutation/ctime and freshness checks passed.
- Publication review found and closed duplicate staging hashing and non-TextGrid injection gaps; final/resume validators remain independent.
- Full candidate suite: 1775 passed, 37 skipped, the same existing JA fixture failure as baseline, 37.72 s.
- Two additional actual-scheduler tests passed after that run: budgets 1 and 8, bounded queue, 8 batches, retained MFA allocation and bounded active weight.
- Three-run alternating local benchmark: normalization median 0.853→0.800 s; work hash reads 14432→9624, raw reads unchanged; 10-batch copy median 5.123→3.973 ms and target reads 55→10. Work/publication bytes and receipt projection match.
- Independent final review: no remaining blocking findings. Details and raw measurements are in the analysis document and benchmark JSON.
- Changes remain uncommitted in the existing workspace. No production model run or production configuration change was performed.
