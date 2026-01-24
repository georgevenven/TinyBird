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
    base = stem
    last_start = None
    last_end = None
    while True:
        match = _CHUNK_MS_RE.match(base)
        if not match:
            break
        base = match.group("base")
        last_start = int(match.group("start"))
        last_end = int(match.group("end"))
    return base, last_start, last_end


def _load_recording_events(annotation_json, bird_id=None):
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
            unit_ids = []
            for unit in event.get("units", []):
                unit_id = unit.get("id")
                if unit_id is None:
                    continue
                unit_ids.append(int(unit_id))
            events_by_stem[stem].append(
                {
                    "onset_ms": event["onset_ms"],
                    "offset_ms": event["offset_ms"],
                    "unit_ids": unit_ids,
                }
            )
    return events_by_stem


def _load_recording_units(annotation_json, bird_id=None, allowed_stems=None):
    if not annotation_json:
        return {}, set()
    data = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    units_by_stem = defaultdict(list)
    all_units = set()
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
            for unit in event.get("units", []):
                unit_id = unit.get("id")
                onset = unit.get("onset_ms")
                offset = unit.get("offset_ms")
                if unit_id is None or onset is None or offset is None:
                    continue
                onset = float(onset)
                offset = float(offset)
                if offset <= onset:
                    continue
                unit_id = int(unit_id)
                units_by_stem[stem].append(
                    {"unit_id": unit_id, "onset_ms": onset, "offset_ms": offset}
                )
                all_units.add(unit_id)
    return units_by_stem, all_units


def _build_unit_index(spec_dir, annotation_json, bird_id=None, allowed_stems=None):
    units_by_stem, all_units = _load_recording_units(
        annotation_json, bird_id=bird_id, allowed_stems=allowed_stems
    )
    if not units_by_stem:
        return defaultdict(list), set()

    unit_events = defaultdict(list)
    for path in Path(spec_dir).glob("*.npy"):
        base_stem, chunk_start, chunk_end = _parse_chunk_ms(path.stem)
        if allowed_stems is not None and base_stem not in allowed_stems:
            continue
        units = units_by_stem.get(base_stem, [])
        if not units:
            continue
        for unit in units:
            onset = unit["onset_ms"]
            offset = unit["offset_ms"]
            if chunk_start is not None:
                if offset <= chunk_start or onset >= chunk_end:
                    continue
                local_onset = max(onset, chunk_start) - chunk_start
                local_offset = min(offset, chunk_end) - chunk_start
            else:
                local_onset = onset
                local_offset = offset

            if local_offset <= local_onset:
                continue

            unit_events[unit["unit_id"]].append(
                {
                    "path": path,
                    "base_stem": base_stem,
                    "chunk_start": chunk_start,
                    "onset_ms": local_onset,
                    "offset_ms": local_offset,
                }
            )

    return unit_events, all_units


def _center_window_on_unit(onset_bin, offset_bin, window_bins, timebins):
    window_bins = min(window_bins, timebins)
    center = (onset_bin + offset_bin) / 2.0
    start = int(round(center - window_bins / 2.0))
    start = max(0, min(start, timebins - window_bins))
    end = start + window_bins
    if onset_bin < start:
        start = max(0, min(onset_bin, timebins - window_bins))
        end = start + window_bins
    if offset_bin > end:
        start = max(0, min(offset_bin - window_bins, timebins - window_bins))
        end = start + window_bins
    return start, end


def _chunk_bounds_to_ms(base_stem, chunk_start, start_bin, end_bin, sr, hop_size):
    chunk_offset = int(chunk_start or 0)
    start_ms = timebins_to_ms(start_bin, sr, hop_size) + chunk_offset
    end_ms = timebins_to_ms(end_bin, sr, hop_size) + chunk_offset
    return base_stem, start_ms, end_ms


def _build_event_index(spec_dir, annotation_json, bird_id=None, allowed_stems=None, mode="classify"):
    events_by_stem = _load_recording_events(annotation_json, bird_id=bird_id)
    if not events_by_stem:
        return [], defaultdict(list), set()

    all_units = set()
    for stem, events in events_by_stem.items():
        if allowed_stems is not None and stem not in allowed_stems:
            continue
        for event in events:
            for unit_id in event.get("unit_ids", []):
                all_units.add(unit_id)

    event_pool = []
    unit_events = defaultdict(list)
    for path in Path(spec_dir).glob("*.npy"):
        base_stem, chunk_start, chunk_end = _parse_chunk_ms(path.stem)
        if allowed_stems is not None and base_stem not in allowed_stems:
            continue
        events = events_by_stem.get(base_stem, [])
        if not events:
            continue
        for event in events:
            if mode in ["classify", "unit_detect"] and not event.get("unit_ids"):
                continue
            onset = event["onset_ms"]
            offset = event["offset_ms"]
            if chunk_start is not None:
                if offset <= chunk_start or onset >= chunk_end:
                    continue
                local_onset = max(onset, chunk_start) - chunk_start
                local_offset = min(offset, chunk_end) - chunk_start
                abs_onset = max(onset, chunk_start)
                abs_offset = min(offset, chunk_end)
            else:
                local_onset = onset
                local_offset = offset
                abs_onset = onset
                abs_offset = offset

            if local_offset <= local_onset:
                continue

            entry = {
                "path": path,
                "base_stem": base_stem,
                "chunk_start": chunk_start,
                "onset_ms": local_onset,
                "offset_ms": local_offset,
                "abs_onset_ms": abs_onset,
                "abs_offset_ms": abs_offset,
                "unit_ids": event.get("unit_ids", []),
            }
            event_pool.append(entry)
            for unit_id in entry["unit_ids"]:
                unit_events[unit_id].append(entry)

    return event_pool, unit_events, all_units


def iter_files(spec_dir, order_file, seed, allowed_stems=None):
    def is_allowed(path):
        if allowed_stems is None:
            return True
        base_stem, _, _ = _parse_chunk_ms(path.stem)
        return base_stem in allowed_stems

    if order_file:
        with open(order_file, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        for name in names:
            path = spec_dir / name
            if path.exists():
                if is_allowed(path):
                    yield path
    else:
        files = sorted(spec_dir.glob("*.npy"))
        if allowed_stems is not None:
            files = [p for p in files if is_allowed(p)]
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

    event_pool = []
    unit_events = defaultdict(list)
    all_units = set()
    if ensure_units:
        unit_events, all_units = _build_unit_index(
            spec_dir, annotation_json, bird_id=bird_id, allowed_stems=allowed_stems
        )
    elif event_chunks:
        event_pool, unit_events, all_units = _build_event_index(
            spec_dir, annotation_json, bird_id=bird_id, allowed_stems=allowed_stems, mode=mode
        )

    if ensure_units:
        budget_bins = seconds_to_timebins(seconds, sr, hop_size)
        planned = []
        missing_units = []
        length_cache = {}
        for unit_id in sorted(all_units):
            events = unit_events.get(unit_id, [])
            if not events:
                missing_units.append(unit_id)
                continue
            entry = rng.choice(events)
            path = entry["path"]
            if path not in length_cache:
                arr = np.load(path, mmap_mode="r")
                length_cache[path] = int(arr.shape[1])
            timebins = length_cache[path]
            if timebins == 0:
                missing_units.append(unit_id)
                continue

            onset_bin = ms_to_timebins(entry["onset_ms"], sr, hop_size)
            offset_bin = ms_to_timebins(entry["offset_ms"], sr, hop_size)
            onset_bin = max(0, min(onset_bin, timebins - 1))
            offset_bin = max(onset_bin + 1, min(offset_bin, timebins))
            unit_bins = max(1, offset_bin - onset_bin)
            window_bins = unit_bins
            if min_timebins and min_timebins > 0:
                window_bins = max(int(min_timebins), unit_bins)
            window_bins = min(window_bins, timebins)
            if window_bins <= 0:
                missing_units.append(unit_id)
                continue
            planned.append(
                {
                    "path": path,
                    "base_stem": entry["base_stem"],
                    "chunk_start": entry.get("chunk_start") or 0,
                    "timebins": timebins,
                    "onset_bin": onset_bin,
                    "offset_bin": offset_bin,
                    "unit_bins": unit_bins,
                    "window_bins": window_bins,
                }
            )
            used_stems.add(entry["base_stem"])

        if missing_units:
            print(f"Warning: Missing units with no available events/specs: {sorted(set(missing_units))}")
        total_bins = sum(item["window_bins"] for item in planned)
        if total_bins > budget_bins:
            excess_bins = total_bins - budget_bins
            order = list(range(len(planned)))
            rng.shuffle(order)
            for idx in order:
                if excess_bins <= 0:
                    break
                item = planned[idx]
                reducible = item["window_bins"] - item["unit_bins"]
                if reducible <= 0:
                    continue
                reduce_by = min(reducible, excess_bins)
                item["window_bins"] -= reduce_by
                excess_bins -= reduce_by
            if excess_bins > 0:
                print(
                    "Warning: unit coverage exceeds budget even after cropping to unit durations."
                )

        total_bins = 0
        for item in planned:
            path = item["path"]
            arr = np.load(path, mmap_mode="r")
            start_bin, end_bin = _center_window_on_unit(
                item["onset_bin"],
                item["offset_bin"],
                item["window_bins"],
                item["timebins"],
            )
            chunk = np.array(arr[:, start_bin:end_bin], dtype=np.float32)
            base_stem, start_ms, end_ms = _chunk_bounds_to_ms(
                item["base_stem"],
                item["chunk_start"],
                start_bin,
                end_bin,
                sr,
                hop_size,
            )
            out_name = f"{base_stem}__ms_{start_ms}_{end_ms}.npy"
            np.save(out_dir / out_name, chunk)
            total_bins += (end_bin - start_bin)

        if total_bins < budget_bins:
            for path in iter_files(spec_dir, order_file, seed, allowed_stems=allowed_stems):
                base_stem, chunk_start, _ = _parse_chunk_ms(path.stem)
                if base_stem in used_stems:
                    continue
                arr = np.load(path, mmap_mode="r")
                timebins = int(arr.shape[1])
                if timebins == 0:
                    continue
                if total_bins + timebins <= budget_bins:
                    copy_or_move(path, out_dir / path.name, move)
                    total_bins += timebins
                    used_stems.add(base_stem)
                    if total_bins >= budget_bins:
                        break
                    continue

                if not truncate_last:
                    break

                remainder_bins = budget_bins - total_bins
                if remainder_bins <= 0:
                    break
                remainder_bins = min(timebins, remainder_bins)
                if remainder_bins >= timebins:
                    copy_or_move(path, out_dir / path.name, move)
                    total_bins += timebins
                    break

                if random_crop and remainder_bins < timebins:
                    start_bin = rng.randint(0, max(0, timebins - remainder_bins))
                else:
                    start_bin = 0
                end_bin = start_bin + remainder_bins
                chunk = np.array(arr[:, start_bin:end_bin], dtype=np.float32)
                base_stem, start_ms, end_ms = _chunk_bounds_to_ms(
                    base_stem, chunk_start, start_bin, end_bin, sr, hop_size
                )
                out_name = f"{base_stem}__ms_{start_ms}_{end_ms}.npy"
                np.save(out_dir / out_name, chunk)

                if move:
                    path.unlink()

                total_bins += (end_bin - start_bin)
                break

        return total_bins * hop_size / sr

    if event_chunks:
        used_event_keys = set()
        rng.shuffle(event_pool)
        for event in event_pool:
            if total_seconds >= seconds:
                break
            event_key = (event["base_stem"], event["abs_onset_ms"], event["abs_offset_ms"])
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
            start_ms = int(event["abs_onset_ms"])
            end_ms = int(event["abs_offset_ms"])
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
        base_stem, chunk_start, _ = _parse_chunk_ms(path.stem)
        base_stem, start_ms, end_ms = _chunk_bounds_to_ms(
            base_stem, chunk_start, start_bin, end_bin, sr, hop_size
        )
        out_name = f"{base_stem}__ms_{start_ms}_{end_ms}.npy"
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
