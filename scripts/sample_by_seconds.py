import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
import re

import numpy as np


def load_audio_params(spec_dir):
    audio_path = Path(spec_dir) / "audio_params.json"
    if not audio_path.exists():
        raise SystemExit(f"audio_params.json not found in {spec_dir}")
    with open(audio_path, "r") as f:
        audio = json.load(f)
    for key in ("sr", "hop_size"):
        if key not in audio:
            raise SystemExit(f"Missing {key} in {audio_path}")
    return audio


def seconds_to_timebins(seconds, sr, hop_size):
    return max(1, int(round(float(seconds) * sr / hop_size)))


def timebins_to_ms(timebins, sr, hop_size):
    return int(round(timebins * hop_size / sr * 1000.0))


def ms_to_timebins(ms, sr, hop_size):
    return int(round(float(ms) / 1000.0 * sr / hop_size))


def _load_allowed_stems(annotation_json, bird_id):
    if not annotation_json:
        return None
    data = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    stems = set()
    for rec in data.get("recordings", []):
        if bird_id and rec.get("recording", {}).get("bird_id") != bird_id:
            continue
        fname = rec.get("recording", {}).get("filename")
        if not fname:
            continue
        stems.add(Path(fname).stem)
    return stems


_CHUNK_MS_RE = re.compile(r"^(?P<base>.+)__ms_(?P<start>\d+)_(?P<end>\d+)$")


def _parse_chunk_ms(stem):
    match = _CHUNK_MS_RE.match(stem)
    if not match:
        return stem, None, None
    return match.group("base"), int(match.group("start")), int(match.group("end"))


def _load_events_by_stem(annotation_json, bird_id=None):
    if not annotation_json:
        return {}
    data = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    events_by_stem = defaultdict(list)
    for rec in data.get("recordings", []):
        if bird_id and rec.get("recording", {}).get("bird_id") != bird_id:
            continue
        fname = rec.get("recording", {}).get("filename")
        if not fname:
            continue
        stem = Path(fname).stem
        for event in rec.get("detected_events", []):
            for unit in event.get("units", []):
                unit_id = unit.get("id")
                if unit_id is None:
                    continue
                events_by_stem[stem].append(
                    {"id": int(unit_id), "onset_ms": unit["onset_ms"], "offset_ms": unit["offset_ms"]}
                )
    return events_by_stem


def _build_unit_index(spec_dir, annotation_json, bird_id=None, allowed_stems=None, mode="classify"):
    if mode not in ["classify", "unit_detect"]:
        return {}, set()
    events_by_stem = _load_events_by_stem(annotation_json, bird_id=bird_id)
    if not events_by_stem:
        return {}, set()
    all_units = set()
    for stem, events in events_by_stem.items():
        if allowed_stems is not None and stem not in allowed_stems:
            continue
        for event in events:
            all_units.add(event["id"])
    events_by_unit = defaultdict(list)
    for path in Path(spec_dir).glob("*.npy"):
        base_stem, chunk_start, chunk_end = _parse_chunk_ms(path.stem)
        if allowed_stems is not None and base_stem not in allowed_stems:
            continue
        events = events_by_stem.get(base_stem, [])
        if not events:
            continue
        for event in events:
            onset = event["onset_ms"]
            offset = event["offset_ms"]
            if chunk_start is not None:
                if offset <= chunk_start or onset >= chunk_end:
                    continue
                local_onset = max(onset, chunk_start) - chunk_start
                local_offset = min(offset, chunk_end) - chunk_start
            else:
                local_onset = onset
                local_offset = offset
            events_by_unit[event["id"]].append(
                {
                    "path": path,
                    "base_stem": base_stem,
                    "chunk_start": chunk_start,
                    "onset_ms": local_onset,
                    "offset_ms": local_offset,
                }
            )
    return events_by_unit, all_units


def _build_detect_events(spec_dir, annotation_json, bird_id=None, allowed_stems=None):
    if not annotation_json:
        return []
    data = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    events_by_stem = defaultdict(list)
    for rec in data.get("recordings", []):
        if bird_id and rec.get("recording", {}).get("bird_id") != bird_id:
            continue
        fname = rec.get("recording", {}).get("filename")
        if not fname:
            continue
        stem = Path(fname).stem
        if allowed_stems is not None and stem not in allowed_stems:
            continue
        for event in rec.get("detected_events", []):
            events_by_stem[stem].append(
                {"onset_ms": event["onset_ms"], "offset_ms": event["offset_ms"]}
            )

    event_pool = []
    for path in Path(spec_dir).glob("*.npy"):
        base_stem, chunk_start, chunk_end = _parse_chunk_ms(path.stem)
        if allowed_stems is not None and base_stem not in allowed_stems:
            continue
        events = events_by_stem.get(base_stem, [])
        if not events:
            continue
        for event in events:
            onset = event["onset_ms"]
            offset = event["offset_ms"]
            if chunk_start is not None:
                if offset <= chunk_start or onset >= chunk_end:
                    continue
                local_onset = max(onset, chunk_start) - chunk_start
                local_offset = min(offset, chunk_end) - chunk_start
            else:
                local_onset = onset
                local_offset = offset
            event_pool.append(
                {
                    "path": path,
                    "base_stem": base_stem,
                    "chunk_start": chunk_start,
                    "onset_ms": local_onset,
                    "offset_ms": local_offset,
                }
            )
    return event_pool


def iter_files(spec_dir, order_file, seed, allowed_stems=None):
    if order_file:
        with open(order_file, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        for name in names:
            path = spec_dir / name
            if path.exists():
                if allowed_stems is None or path.stem in allowed_stems:
                    yield path
    else:
        if allowed_stems is None:
            files = sorted(spec_dir.glob("*.npy"))
        else:
            files = [spec_dir / f"{stem}.npy" for stem in sorted(allowed_stems)]
            files = [p for p in files if p.exists()]
        rng = random.Random(seed)
        rng.shuffle(files)
        for path in files:
            yield path


def copy_or_move(src, dst, move):
    if move:
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


def sample_by_seconds(
    spec_dir,
    out_dir,
    *,
    seconds,
    seed,
    order_file=None,
    truncate_last=True,
    move=False,
    annotation_json=None,
    bird_id=None,
    ensure_units=False,
    min_timebins=0,
    mode="classify",
    random_crop=False,
    event_chunks=False,
):
    spec_dir = Path(spec_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_params = load_audio_params(spec_dir)
    sr = int(audio_params["sr"])
    hop_size = int(audio_params["hop_size"])
    rng = random.Random(seed)

    audio_params_path = spec_dir / "audio_params.json"
    if audio_params_path.exists():
        shutil.copy2(audio_params_path, out_dir / "audio_params.json")

    if bird_id and not annotation_json:
        raise SystemExit("bird_id requires annotation_json")
    if ensure_units and not annotation_json:
        raise SystemExit("ensure_units requires annotation_json")
    if event_chunks and not annotation_json:
        raise SystemExit("event_chunks requires annotation_json")

    allowed_stems = _load_allowed_stems(annotation_json, bird_id)
    used_stems = set()
    total_seconds = 0.0

    events_by_unit = {}
    all_units = set()
    event_pool = []
    if ensure_units or (event_chunks and mode in ["classify", "unit_detect"]):
        events_by_unit, all_units = _build_unit_index(
            spec_dir, annotation_json, bird_id=bird_id, allowed_stems=allowed_stems, mode=mode
        )
        if event_chunks:
            for events in events_by_unit.values():
                event_pool.extend(events)

    if event_chunks and mode == "detect":
        event_pool = _build_detect_events(
            spec_dir, annotation_json, bird_id=bird_id, allowed_stems=allowed_stems
        )

    used_event_keys = set()
    if ensure_units:
        missing_units = []
        for unit_id in sorted(all_units):
            events = events_by_unit.get(unit_id, [])
            if not events:
                missing_units.append(unit_id)
                continue
            event = rng.choice(events)
            event_key = (str(event["path"]), event["onset_ms"], event["offset_ms"])
            used_event_keys.add(event_key)
            path = event["path"]
            arr = np.load(path, mmap_mode="r")
            timebins = int(arr.shape[1])
            if timebins == 0:
                missing_units.append(unit_id)
                continue

            onset_bin = ms_to_timebins(event["onset_ms"], sr, hop_size)
            offset_bin = ms_to_timebins(event["offset_ms"], sr, hop_size)
            onset_bin = max(0, min(onset_bin, timebins - 1))
            offset_bin = max(onset_bin + 1, min(offset_bin, timebins))

            if event_chunks:
                start_bin = onset_bin
                end_bin = offset_bin
            else:
                window_bins = int(min_timebins) if min_timebins and min_timebins > 0 else timebins
                window_bins = min(window_bins, timebins)
                if window_bins <= 0:
                    missing_units.append(unit_id)
                    continue
                if window_bins >= timebins:
                    start_bin = 0
                else:
                    start_min = max(0, offset_bin - window_bins)
                    start_max = min(onset_bin, timebins - window_bins)
                    if start_min > start_max:
                        start_bin = max(0, min(onset_bin, timebins - window_bins))
                    else:
                        start_bin = rng.randint(start_min, start_max)
                end_bin = start_bin + window_bins
                if end_bin > timebins:
                    end_bin = timebins
                    start_bin = max(0, end_bin - window_bins)

            chunk = np.array(arr[:, start_bin:end_bin], dtype=np.float32)
            base_stem = event["base_stem"]
            start_ms = timebins_to_ms(start_bin, sr, hop_size)
            end_ms = timebins_to_ms(end_bin, sr, hop_size)
            chunk_start = event.get("chunk_start")
            if chunk_start is not None:
                start_ms += chunk_start
                end_ms += chunk_start
            out_name = f"{base_stem}__ms_{start_ms}_{end_ms}.npy"
            np.save(out_dir / out_name, chunk)
            used_stems.add(base_stem)
            total_seconds += (end_bin - start_bin) * hop_size / sr

        if missing_units:
            print(f"Warning: Missing units with no available events/specs: {sorted(set(missing_units))}")
        if total_seconds > seconds:
            print(
                f"Warning: unit coverage uses {total_seconds:.3f}s which exceeds target {seconds:.3f}s"
            )

    if event_chunks:
        rng.shuffle(event_pool)
        for event in event_pool:
            if total_seconds >= seconds:
                break
            event_key = (str(event["path"]), event["onset_ms"], event["offset_ms"])
            if event_key in used_event_keys:
                continue
            path = event["path"]
            arr = np.load(path, mmap_mode="r")
            timebins = int(arr.shape[1])
            if timebins == 0:
                continue
            onset_bin = ms_to_timebins(event["onset_ms"], sr, hop_size)
            offset_bin = ms_to_timebins(event["offset_ms"], sr, hop_size)
            onset_bin = max(0, min(onset_bin, timebins - 1))
            offset_bin = max(onset_bin + 1, min(offset_bin, timebins))
            start_bin = onset_bin
            end_bin = offset_bin
            duration_seconds = (end_bin - start_bin) * hop_size / sr
            if total_seconds + duration_seconds > seconds + 1e-9:
                continue
            chunk = np.array(arr[:, start_bin:end_bin], dtype=np.float32)
            base_stem = event["base_stem"]
            start_ms = timebins_to_ms(start_bin, sr, hop_size)
            end_ms = timebins_to_ms(end_bin, sr, hop_size)
            chunk_start = event.get("chunk_start")
            if chunk_start is not None:
                start_ms += chunk_start
                end_ms += chunk_start
            out_name = f"{base_stem}__ms_{start_ms}_{end_ms}.npy"
            np.save(out_dir / out_name, chunk)
            used_event_keys.add(event_key)
            total_seconds += duration_seconds
        return total_seconds

    for path in iter_files(spec_dir, order_file, seed, allowed_stems=allowed_stems):
        if ensure_units and path.stem in used_stems:
            continue
        arr = np.load(path, mmap_mode="r")
        timebins = int(arr.shape[1])
        duration_seconds = timebins * hop_size / sr

        if total_seconds + duration_seconds <= seconds + 1e-9:
            copy_or_move(path, out_dir / path.name, move)
            total_seconds += duration_seconds
            if total_seconds >= seconds:
                break
            continue

        remaining = seconds - total_seconds
        if remaining <= 0:
            break
        if not truncate_last:
            break

        remainder_bins = seconds_to_timebins(remaining, sr, hop_size)
        if min_timebins and remainder_bins < int(min_timebins):
            remainder_bins = int(min_timebins)
        remainder_bins = min(timebins, remainder_bins)
        if remainder_bins >= timebins:
            copy_or_move(path, out_dir / path.name, move)
            total_seconds += duration_seconds
            break

        if random_crop and remainder_bins < timebins:
            start_bin = rng.randint(0, max(0, timebins - remainder_bins))
        else:
            start_bin = 0
        end_bin = min(timebins, start_bin + remainder_bins)
        chunk = np.array(arr[:, start_bin:end_bin], dtype=np.float32)
        start_ms = timebins_to_ms(start_bin, sr, hop_size)
        end_ms = timebins_to_ms(end_bin, sr, hop_size)
        out_name = f"{path.stem}__ms_{start_ms}_{end_ms}.npy"
        np.save(out_dir / out_name, chunk)

        if move:
            path.unlink()

        total_seconds += (end_bin - start_bin) * hop_size / sr
        break

    return total_seconds


def main():
    parser = argparse.ArgumentParser(description="Sample spectrogram files to a target duration.")
    parser.add_argument("--spec_dir", required=True, help="Input directory with .npy spectrograms")
    parser.add_argument("--out_dir", required=True, help="Output directory for sampled files")
    parser.add_argument("--seconds", type=float, required=True, help="Target duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--order_file", type=str, default=None, help="Optional order file for deterministic selection")
    parser.add_argument("--annotation_json", type=str, default=None, help="Optional annotations JSON to filter recordings")
    parser.add_argument("--bird_id", type=str, default=None, help="Bird ID to filter (requires --annotation_json)")
    parser.add_argument("--ensure_units", action="store_true", help="ensure at least one instance of each unit (classify only)")
    parser.add_argument("--min_timebins", type=int, default=0, help="minimum timebins for cropped chunks (0 disables)")
    parser.add_argument("--mode", type=str, default="classify", choices=["detect", "classify", "unit_detect"], help="label mode for unit coverage")
    parser.add_argument("--random_crop", action="store_true", help="randomly crop partial files instead of taking from start")
    parser.add_argument("--event_chunks", action="store_true", help="write chunks only for detected events/units")
    parser.add_argument("--truncate_last", dest="truncate_last", action="store_true", help="Truncate last file to hit exact seconds")
    parser.add_argument("--no_truncate_last", dest="truncate_last", action="store_false", help="Do not truncate last file")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.set_defaults(truncate_last=True)
    args = parser.parse_args()

    total = sample_by_seconds(
        args.spec_dir,
        args.out_dir,
        seconds=args.seconds,
        seed=args.seed,
        order_file=args.order_file,
        truncate_last=args.truncate_last,
        move=args.move,
        annotation_json=args.annotation_json,
        bird_id=args.bird_id,
        ensure_units=args.ensure_units,
        min_timebins=args.min_timebins,
        mode=args.mode,
        random_crop=args.random_crop,
        event_chunks=args.event_chunks,
    )
    print(f"Sampled {total:.3f}s to {args.out_dir}")


if __name__ == "__main__":
    main()
