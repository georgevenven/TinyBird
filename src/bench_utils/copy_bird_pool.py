import argparse
import glob
import json
import os
import shutil
from pathlib import Path


def find_spec_matches(spec_dir, base_name):
    patterns = [
        os.path.join(spec_dir, f"{base_name}.*"),
        os.path.join(spec_dir, f"{base_name}__ms_*"),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    return matches


def copy_bird_pool(annotation_file, spec_dir, out_dir, bird_id=None, move_files=False):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(annotation_file, "r") as f:
        data = json.load(f)

    recordings = []
    for rec in data.get("recordings", []):
        rec_bird = rec.get("recording", {}).get("bird_id")
        if bird_id is not None and rec_bird != bird_id:
            continue
        recordings.append(rec)

    if not recordings:
        raise SystemExit(f"No recordings found for bird_id={bird_id}")

    copied = 0
    seen = set()
    for recording in recordings:
        filename = recording.get("recording", {}).get("filename")
        if not filename:
            continue
        base_name = Path(filename).stem
        for src in find_spec_matches(spec_dir, base_name):
            if src in seen:
                continue
            seen.add(src)
            dst = out_path / Path(src).name
            if dst.exists():
                continue
            if move_files:
                shutil.move(src, dst)
            else:
                shutil.copy2(src, dst)
            copied += 1

    audio_params = Path(spec_dir) / "audio_params.json"
    if audio_params.exists():
        shutil.copy2(audio_params, out_path / "audio_params.json")

    filtered = dict(data)
    filtered["recordings"] = recordings
    filtered_path = out_path / "annotations_filtered.json"
    with open(filtered_path, "w") as f:
        json.dump(filtered, f, indent=2)

    action = "Moved" if move_files else "Copied"
    print(
        f"{action} {copied} files from {spec_dir} to {out_path} "
        f"for bird_id={bird_id} ({len(recordings)} recordings)."
    )
    print(f"Wrote filtered annotations: {filtered_path}")


def main():
    parser = argparse.ArgumentParser(description="Copy all spec files for a bird into a pool directory.")
    parser.add_argument("--annotation_file", required=True, help="Path to annotation JSON.")
    parser.add_argument("--spec_dir", required=True, help="Directory of spec .npy files.")
    parser.add_argument("--out_dir", required=True, help="Destination directory for the pool copy.")
    parser.add_argument("--bird_id", default=None, help="Bird ID to filter by.")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying.")
    args = parser.parse_args()

    copy_bird_pool(
        annotation_file=args.annotation_file,
        spec_dir=args.spec_dir,
        out_dir=args.out_dir,
        bird_id=args.bird_id,
        move_files=args.move,
    )


if __name__ == "__main__":
    main()
