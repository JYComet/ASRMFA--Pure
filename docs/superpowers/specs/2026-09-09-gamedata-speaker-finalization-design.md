# GAMEDATA Speaker Finalization Design

## Goal

Finalize the usable outputs for all eight GAMEDATA games, organize accepted
TextGrids by game and original speaker, and place the corresponding edge-silence
normalized WAVs under `/mnt/Raw/GAMESL/<game>/<speaker>/`.

## Accepted source sets

- Six published games consume the flat accepted TextGrids already present in
  `/mnt/Raw/GAMEDATA_对齐_20260903/<game>/`.
- `genshin` consumes the 86,782 accepted TextGrids from
  `/mnt/nvme3/mfa_work_gamedata_genshin_20260903/output_staging/recover_20260908T1420Z/`.
- `reverse1999` consumes the 11,923 accepted TextGrids from the completed
  2026-09-04 postprocess staging directory. The three NVV provenance errors and
  one missing-MFA row are excluded; their combined rate is 0.01009%, so no
  rerun is performed.

## Speaker mapping

The authoritative mapping is each game's `.stage_manifest.json`. The speaker
is the immediate parent directory of the recorded original audio path. If a
stem or usable parent cannot be resolved, it is placed under `default` and
recorded in the receipt.

## Output layout

Accepted TextGrids are stored at
`/mnt/Raw/GAMEDATA_对齐_20260903/<game>/<speaker>/<stem>.TextGrid`.
Corresponding WAVs are stored at
`/mnt/Raw/GAMESL/<game>/<speaker>/<stem>.wav`.

Existing flat published TextGrids are moved within the same game directory;
metadata files remain at the game root. Genshin and reverse1999 TextGrids are
copied from their recovery staging directories. The operation is idempotent and
fails on conflicting destinations.

For games with a workspace `padded_audio/<stem>.wav`, that authoritative padded
WAV is copied to GAMESL. For accepted stems without such an artifact (currently
baijing and reverse1999), the original manifest WAV is normalized to 0.5 seconds
of leading and trailing silence and the destination TextGrid is shifted by the
computed leading-edge offset.

## Safety and verification

Each game is planned before writes. Unsafe speaker names, missing manifest
records, source collisions, missing audio, and conflicting destination files
fail closed. An atomic `.speaker_classification_receipt.json` records counts,
speaker distribution, audio mode, and unresolved stems. Completion requires:

- accepted source count equals classified TextGrid count;
- classified TextGrid stems equal GAMESL WAV stems per game;
- no flat accepted TextGrids remain at game roots;
- every TextGrid parses and every WAV is readable;
- aggregate classified count equals 151,877.

