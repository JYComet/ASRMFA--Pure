import os
import shutil
from pathlib import Path

INPUT_DIR = r"\\RS3621\Research_TTS\Data\Raw\ASR_compare\input"
GROUP_SIZE = 1200

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".opus", ".aiff"}


def main():
    input_path = Path(INPUT_DIR)

    # Collect all audio files with their mtime
    files = []
    for f in input_path.iterdir():
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            files.append((f.stat().st_mtime, f))

    if not files:
        print("No audio files found.")
        return

    # Sort by modification time (oldest first)
    files.sort(key=lambda x: x[0])

    total = len(files)
    num_groups = (total + GROUP_SIZE - 1) // GROUP_SIZE
    print(f"Found {total} audio files, grouping into {num_groups} folder(s) ({GROUP_SIZE} per group).")

    for group_idx in range(num_groups):
        start = group_idx * GROUP_SIZE
        end = min(start + GROUP_SIZE, total)
        group_dir = input_path / str(group_idx + 1)
        group_dir.mkdir(exist_ok=True)

        for _, filepath in files[start:end]:
            dest = group_dir / filepath.name
            shutil.move(str(filepath), str(dest))

        print(f"Group {group_idx + 1}: moved {end - start} files.")

    print("Done.")


if __name__ == "__main__":
    main()
