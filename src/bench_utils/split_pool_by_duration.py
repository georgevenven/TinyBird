import argparse
import glob
import json
import os
import random
import shutil
from pathlib import Path


def calculate_ms(detected_events):
    total_ms = 0
    for event in detected_events:
        total_ms += event["offset_ms"] - event["onset_ms"]
    return total_ms


def find_spec_matches(spec_dir, base_name):
    patterns = [
        os.path.join(spec_dir, f"{base_name}.*"),
        os.path.join(spec_dir, f"{base_name}__ms_*"),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    return matches


def split_pool_by_duration(pool_dir, test_dir, annotation_json, train_percent=80, seed=42):
    with open(annotation_json, "r") as f:
        data = json.load(f)

    recordings = data.get("recordings", [])
    if not recordings:
        print("No recordings found in annotations.")
        return

    recs = []
    total_ms = 0.0
    for rec in recordings:
        ms = calculate_ms(rec.get("detected_events", []))
        recs.append((rec, ms))
        total_ms += ms

    random.seed(seed)
    random.shuffle(recs)

    target_train_ms = total_ms * (train_percent / 100.0)
    train_ms = 0.0
    test_recs = []
    for rec, ms in recs:
        if train_ms < target_train_ms:
            train_ms += ms
        else:
            test_recs.append(rec)

    os.makedirs(test_dir, exist_ok=True)

    moved = 0
    seen = set()
    for rec in test_recs:
        filename = rec.get("recording", {}).get("filename")
        if not filename:
            continue
        base_name = Path(filename).stem
        for src in find_spec_matches(pool_dir, base_name):
            if src in seen:
                continue
            seen.add(src)
            dst = os.path.join(test_dir, os.path.basename(src))
            if os.path.exists(dst):
                continue
            shutil.move(src, dst)
            moved += 1

    audio_params = Path(pool_dir) / "audio_params.json"
    if audio_params.exists():
        shutil.copy2(audio_params, Path(test_dir) / "audio_params.json")

    print(
        f"Moved {moved} files to {test_dir} "
        f"(train_ms={train_ms:.2f}, total_ms={total_ms:.2f})."
    )


def main():
    parser = argparse.ArgumentParser(description="Split a pool directory into train/test by duration.")
    parser.add_argument("--pool_dir", required=True, help="Directory containing the full pool.")
    parser.add_argument("--test_dir", required=True, help="Destination directory for test files.")
    parser.add_argument("--annotation_json", required=True, help="Filtered annotations for the pool.")
    parser.add_argument("--train_percent", type=float, default=80.0, help="Percent of duration to keep in pool.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    split_pool_by_duration(
        pool_dir=args.pool_dir,
        test_dir=args.test_dir,
        annotation_json=args.annotation_json,
        train_percent=args.train_percent,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
