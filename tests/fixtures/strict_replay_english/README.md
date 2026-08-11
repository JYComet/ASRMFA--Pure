# Strict-replay English current-v2 fixtures

These are metadata-only, synthetic fail-closed cases.  They intentionally do
not contain production WAVs, TextGrids, MFA labels, or provenance timestamps.
The canonical 96-slot manifest remains an external authority; tests copy it to
a temporary directory and then apply the mutations listed in `cases.json`.

Run the verifier against a freshly imported pilot receipt with:

```text
python scripts/verify_strict_replay_english_subset.py \
  /tmp/<strict-replay-run>/output/strict_replay_import.json
```

Each mutation must return exit code 1.  The fixture names correspond to the
fail-closed boundaries implemented by the verifier:

* `historical_v1_as_v2` changes the subset/ledger schema to `strict-en-mfa-v1`;
* `global_hash_mismatch` changes the canonical selected-membership digest;
* `canonical_external_stem` appends a ledger for a stem not in the selected
  24-slot pilot;
* `import_receipt_hash_mismatch` detaches the subset from its import receipt;
* `ledger_hash_mismatch` changes a declared ledger digest without changing the
  file.

