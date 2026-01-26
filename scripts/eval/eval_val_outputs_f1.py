#!/usr/bin/env python3
"""
Compute F1 and FER from exported val_outputs for finetuned vs linear_probe runs.

Mirrors supervised_train.py compute_f1_score:
  - Binary (num_classes == 2): positive-class F1 with sigmoid threshold 0.5
  - Multi-class: macro-F1 over all classes (including silence=0)

FER is computed as the percent misclassified timesteps (patches) using argmax
over logits.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# Add `src/` to path because internal modules use absolute imports like `import utils`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from utils import parse_chunk_ms, clip_labels_to_chunk


def _round_ms(x: float) -> int:
    return int(round(float(x)))


def _load_annotations(path: str, mode: str) -> Dict[str, List[Tuple[int, int, int]]]:
    """
    Return mapping: filename -> list of (onset_ms, offset_ms, class_id).
    For detect: class_id is 1 for any event.
    For unit_detect/classify: class_id is unit["id"] + 1 (silence=0).
    """
    with open(path, "r") as f:
        data = json.load(f)

    out: Dict[str, List[Tuple[int, int, int]]] = {}
    for rec in data.get("recordings", []):
        filename = Path(rec["recording"]["filename"]).stem
        items: List[Tuple[int, int, int]] = []

        if mode == "detect":
            for event in rec.get("detected_events", []):
                onset = event.get("onset_ms")
                offset = event.get("offset_ms")
                if onset is None or offset is None:
                    continue
                items.append((_round_ms(onset), _round_ms(offset), 1))
        else:
            for event in rec.get("detected_events", []):
                for unit in event.get("units", []):
                    onset = unit.get("onset_ms")
                    offset = unit.get("offset_ms")
                    if onset is None or offset is None:
                        continue
                    class_id = int(unit.get("id", 0)) + 1
                    items.append((_round_ms(onset), _round_ms(offset), class_id))

        items.sort(key=lambda x: x[0])
        out[filename] = items

    return out


def _labels_from_units(
    units: List[Tuple[int, int, int]],
    total_timebins: int,
    ms_per_timebin: float,
    mode: str,
) -> np.ndarray:
    labels = np.zeros((total_timebins,), dtype=np.int64)
    if not units:
        return labels

    for onset_ms, offset_ms, class_id in units:
        if offset_ms <= onset_ms:
            continue

        # timebin i maps to ms = (i + 1) * ms_per_timebin
        start_i = max(0, int(math.ceil(onset_ms / ms_per_timebin)) - 1)
        end_i = int(math.floor((offset_ms - 1) / ms_per_timebin)) - 1
        if end_i < start_i:
            continue
        end_i = min(end_i, total_timebins - 1)

        if mode == "classify":
            labels[start_i : end_i + 1] = class_id
        else:
            labels[start_i : end_i + 1] = 1

    return labels

def _infer_probe_mode(run_name: str, run_config: Dict[str, object]) -> str:
    if int(run_config.get("lora_rank", 0) or 0) > 0:
        return "lora"
    name = run_name.lower()
    if name.startswith("lora") or name.startswith("lora_") or name.startswith("lora/"):
        return "lora"
    if name.startswith("linear") or name.startswith("linear_") or name.startswith("linear/"):
        return "linear"
    if name.startswith("finetune") or name.startswith("finetune_") or name.startswith("finetune/"):
        return "finetune"
    return "unknown"


def _load_run_config(run_dir: Path) -> Dict[str, object]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_run_path(path_fragment: str) -> Path:
    if not path_fragment:
        return Path()
    if os.path.isabs(path_fragment):
        return Path(path_fragment)
    project_relative = (PROJECT_ROOT / path_fragment).resolve()
    if project_relative.exists():
        return project_relative
    return (PROJECT_ROOT / "runs" / path_fragment).resolve()


def _load_pretrained_config(run_config: Dict[str, object]) -> Dict[str, object]:
    pretrained_path = _resolve_run_path(str(run_config.get("pretrained_run") or ""))
    if not pretrained_path:
        return {}
    config_path = pretrained_path / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _frozen_layers(run_config: Dict[str, object], pretrained_config: Dict[str, object]) -> str:
    num_layers = pretrained_config.get("enc_n_layer")
    try:
        num_layers = int(num_layers) if num_layers is not None else None
    except (TypeError, ValueError):
        num_layers = None

    lora_rank = int(run_config.get("lora_rank", 0) or 0)
    freeze_encoder = bool(run_config.get("freeze_encoder", False))
    freeze_up_to = run_config.get("freeze_encoder_up_to", None)

    if lora_rank > 0 or freeze_encoder:
        if num_layers is None:
            return "all"
        return ",".join(str(i) for i in range(num_layers))

    if freeze_up_to is None:
        return "none"

    try:
        idx = int(freeze_up_to)
    except (TypeError, ValueError):
        return "unknown"

    if num_layers is not None:
        if idx < 0:
            idx = num_layers + idx
        if idx < 0:
            return "none"
        idx = min(idx, num_layers - 1)
    if idx < 0:
        return "none"
    return ",".join(str(i) for i in range(idx + 1))


def _build_label_index(annotations: Dict[str, object], mode: str) -> Dict[str, list]:
    if mode not in ["detect", "classify", "unit_detect"]:
        raise ValueError("mode must be 'detect', 'classify', or 'unit_detect'")

    label_index: Dict[str, list] = {}
    for rec in annotations.get("recordings", []):
        rec_filename = Path(rec["recording"]["filename"]).stem
        events = rec.get("detected_events", [])
        if mode == "detect":
            labels = [
                {"onset_ms": event["onset_ms"], "offset_ms": event["offset_ms"]}
                for event in events
            ]
        else:
            labels = [unit for event in events for unit in event.get("units", [])]
        label_index[rec_filename] = labels
    return label_index


def _unique_classes_in_dir(
    spec_dir: Path,
    label_index: Dict[str, list],
    mode: str,
) -> set:
    classes: set = set()
    files = sorted(spec_dir.glob("*.npy"))
    if files:
        classes.add(0)
    for path in files:
        base_filename, chunk_start_ms, chunk_end_ms = parse_chunk_ms(path.stem)
        labels = label_index.get(base_filename)
        if labels is None:
            continue
        labels = clip_labels_to_chunk(labels, chunk_start_ms, chunk_end_ms)
        if not labels:
            continue
        if mode in ["detect", "unit_detect"]:
            classes.add(1)
        else:
            for label in labels:
                if "id" in label:
                    classes.add(int(label["id"]) + 1)
    return classes


def _parse_train_dir_metadata(train_dir: str) -> Tuple[str, str]:
    if not train_dir:
        return "", ""
    path = Path(train_dir)
    parts = [p for p in path.parts if p]
    task = None
    task_idx = None
    for marker in ("classify", "unit_detect", "detect"):
        if marker in parts:
            task = marker
            task_idx = parts.index(marker)
            break
    bird = ""
    if task == "detect":
        bird = "all"
    elif task_idx is not None and task_idx + 1 < len(parts):
        bird = parts[task_idx + 1]

    seconds = ""
    match = re.match(r"train_(?P<seconds>.+)s$", path.name)
    if match:
        seconds = match.group("seconds").replace("p", ".")
    return bird, seconds


def _iter_run_dirs(runs_root: Path) -> Iterable[Tuple[str, Path]]:
    flat_runs: List[Path] = []
    probe_dirs: List[Path] = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "val_outputs").exists() or (child / "config.json").exists():
            flat_runs.append(child)
        else:
            probe_dirs.append(child)

    for run_dir in flat_runs:
        run_config = _load_run_config(run_dir)
        yield _infer_probe_mode(run_dir.name, run_config), run_dir

    for probe_dir in sorted(probe_dirs):
        probe_name = probe_dir.name
        for run_dir in sorted(probe_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_config = _load_run_config(run_dir)
            if run_config:
                yield _infer_probe_mode(run_dir.name, run_config), run_dir
            else:
                yield probe_name, run_dir


def _valid_patches(length: int, patch_width: int) -> int:
    if length <= 0:
        return 0
    return int(math.ceil(length / patch_width))


def _f1_binary(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1 * 100.0


def _f1_macro(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    f1s: List[float] = []
    for c in range(num_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) * 100.0 if f1s else 0.0


def _species_from_run(run_dir: Path) -> str:
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            annot = Path(cfg.get("annotation_file", "")).name.lower()
            if "canary" in annot:
                return "Canary"
            if annot.startswith("bf") or "bengalese" in annot:
                return "Bengalese_Finch"
            if annot.startswith("zf") or "zebra" in annot:
                return "Zebra_Finch"
        except Exception:
            pass

    for sp in ["Bengalese_Finch", "Canary", "Zebra_Finch"]:
        if run_dir.name.startswith(sp):
            return sp
    return "unknown"


def _eval_val_outputs_ms(
    val_dir: Path,
    annotation_json: Path,
    mode: str,
    audio_params: Path,
    threshold: float,
) -> Dict[str, float]:
    logits = np.load(val_dir / "logits.npy", mmap_mode="r")
    window_starts = np.load(val_dir / "window_starts.npy", mmap_mode="r")
    window_lengths = np.load(val_dir / "window_lengths.npy", mmap_mode="r")
    with open(val_dir / "filenames.json", "r") as f:
        filenames = json.load(f)

    with open(val_dir / "meta.json", "r") as f:
        meta = json.load(f)

    num_classes = int(meta["num_classes"])
    patch_width = int(meta["patch_width"])
    mode = mode or meta.get("mode", "unknown")

    annotations = _load_annotations(str(annotation_json), mode)

    with open(audio_params, "r") as f:
        audio = json.load(f)
    if "sr" not in audio or "hop_size" not in audio:
        raise SystemExit(f"audio_params missing sr/hop_size: {audio_params}")
    ms_per_timebin = float(audio["hop_size"]) / float(audio["sr"]) * 1000.0

    file_max_timebin: Dict[str, int] = {}
    for fn, start, length in zip(filenames, window_starts, window_lengths):
        end = int(start) + int(length)
        file_max_timebin[fn] = max(file_max_timebin.get(fn, 0), end)

    labels_by_file: Dict[str, np.ndarray] = {}
    for fn, max_t in file_max_timebin.items():
        base_filename, chunk_start_ms, chunk_end_ms = parse_chunk_ms(fn)
        units = annotations.get(base_filename, [])
        if chunk_start_ms is not None:
            clipped = []
            chunk_end = chunk_end_ms if chunk_end_ms is not None else float("inf")
            for onset_ms, offset_ms, class_id in units:
                if offset_ms <= chunk_start_ms or onset_ms >= chunk_end:
                    continue
                new_onset = max(onset_ms, chunk_start_ms) - chunk_start_ms
                new_offset = min(offset_ms, chunk_end) - chunk_start_ms
                clipped.append((new_onset, new_offset, class_id))
            units = clipped
        labels_by_file[fn] = _labels_from_units(units, max_t, ms_per_timebin, mode)

    all_preds = []
    all_labels = []

    for i in range(logits.shape[0]):
        fn = filenames[i]
        start = int(window_starts[i])
        length = int(window_lengths[i])

        if mode in ["detect", "unit_detect"]:
            probs = 1.0 / (1.0 + np.exp(-logits[i, :, 0]))
            pred_patches = (probs > threshold).astype(np.int64)
        else:
            pred_patches = np.argmax(logits[i, :, :], axis=-1).astype(np.int64)

        pred_time = np.repeat(pred_patches, patch_width)[:length]
        labels_full = labels_by_file.get(fn, np.zeros((start + length,), dtype=np.int64))
        label_time = labels_full[start : start + length]

        all_preds.append(pred_time)
        all_labels.append(label_time)

    all_preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    all_labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)

    if mode in ["detect", "unit_detect"]:
        f1 = _f1_binary(all_preds, all_labels)
    else:
        f1 = _f1_macro(all_preds, all_labels, num_classes)

    correct = int((all_preds == all_labels).sum()) if all_preds.size else 0
    total = int(all_labels.size)
    fer = 100.0 * (1.0 - (correct / total)) if total > 0 else 0.0

    return {
        "f1": f1,
        "fer": fer,
        "num_classes": float(num_classes),
        "patch_width": float(patch_width),
        "mode": mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute F1 from val_outputs across finetuned/linear_probe runs."
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="/media/george-vengrovski/Desk SSD/results",
        help="Root folder containing runs/finetuned and runs/linear_probe",
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default=None,
        help="Optional direct path to runs directory (supports flat or probe-mode subdirs).",
    )
    parser.add_argument(
        "--run_names",
        type=str,
        default=None,
        help="Comma-separated run directory names to evaluate (optional filter).",
    )
    parser.add_argument(
        "--run_names_file",
        type=str,
        default=None,
        help="File containing run directory names to evaluate (one per line).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path (default: <results_root>/eval_f1.csv)",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Output summary CSV path (default: <results_root>/eval_f1_summary.csv)",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary F1")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    runs_root = Path(args.runs_root) if args.runs_root else (results_root / "runs")
    out_csv = Path(args.out_csv) if args.out_csv else results_root / "eval_f1.csv"
    summary_csv = (
        Path(args.summary_csv) if args.summary_csv else results_root / "eval_f1_summary.csv"
    )

    header = [
        "f1",
        "fer",
        "probe_mode",
        "mode",
        "species",
        "bird",
        "train_seconds",
        "num_classes",
        "num_classes_train",
        "num_classes_val",
        "patch_width",
        "frozen_layers",
        "steps",
        "lr",
        "batch_size",
        "class_weighting",
        "run_name",
        "created_at",
    ]
    rows: List[Dict[str, object]] = []
    records: List[Dict[str, object]] = []

    run_filter: set[str] = set()
    if args.run_names:
        run_filter.update({n.strip() for n in args.run_names.split(",") if n.strip()})
    if args.run_names_file:
        run_names_path = Path(args.run_names_file)
        if not run_names_path.exists():
            raise SystemExit(f"Run names file not found: {run_names_path}")
        for line in run_names_path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name:
                run_filter.add(name)

    for probe_mode, run_dir in _iter_run_dirs(runs_root):
        rel_name = ""
        try:
            rel_name = run_dir.relative_to(runs_root).as_posix()
        except ValueError:
            rel_name = run_dir.name
        if run_filter and run_dir.name not in run_filter and rel_name not in run_filter:
            continue
        val_dir = run_dir / "val_outputs"
        if not val_dir.exists():
            continue
        run_config = _load_run_config(run_dir)
        pretrained_config = _load_pretrained_config(run_config)
        annotation_path = run_config.get("annotation_file")
        if not annotation_path:
            raise SystemExit(f"Missing annotation_file in config for run: {run_dir}")

        annot_path = Path(annotation_path)
        if not annot_path.is_absolute():
            annot_path = (PROJECT_ROOT / annot_path).resolve()
        if not annot_path.exists():
            raise SystemExit(f"Annotation file not found: {annot_path}")

        val_dir_cfg = run_config.get("val_dir")
        train_dir_cfg = run_config.get("train_dir")
        audio_params_path = None
        if val_dir_cfg:
            candidate = Path(val_dir_cfg)
            if not candidate.is_absolute():
                candidate = (PROJECT_ROOT / candidate).resolve()
            candidate = candidate / "audio_params.json"
            if candidate.exists():
                audio_params_path = candidate
        if audio_params_path is None and train_dir_cfg:
            candidate = Path(train_dir_cfg)
            if not candidate.is_absolute():
                candidate = (PROJECT_ROOT / candidate).resolve()
            candidate = candidate / "audio_params.json"
            if candidate.exists():
                audio_params_path = candidate
        if audio_params_path is None:
            raise SystemExit(f"audio_params.json not found for run: {run_dir}")

        stats = _eval_val_outputs_ms(val_dir, annot_path, str(run_config.get("mode") or ""), audio_params_path, args.threshold)
        species = _species_from_run(run_dir)
        frozen_layers = _frozen_layers(run_config, pretrained_config)
        steps = run_config.get("steps", "")
        lr = run_config.get("lr", "")
        batch_size = run_config.get("batch_size", "")
        class_weighting = int(bool(run_config.get("class_weighting", False)))

        mode = run_config.get("mode") or stats.get("mode") or "unknown"
        annotation_path = run_config.get("annotation_file")
        train_dir = run_config.get("train_dir")
        val_dir_cfg = run_config.get("val_dir")
        bird, train_seconds = _parse_train_dir_metadata(train_dir)
        if not bird and mode == "detect":
            bird = "all"
        num_classes_train = 0
        num_classes_val = 0
        if annotation_path and train_dir and val_dir_cfg:
            annot_path = Path(annotation_path)
            if not annot_path.is_absolute():
                annot_path = (PROJECT_ROOT / annot_path).resolve()
            train_path = Path(train_dir)
            if not train_path.is_absolute():
                train_path = (PROJECT_ROOT / train_path).resolve()
            val_path = Path(val_dir_cfg)
            if not val_path.is_absolute():
                val_path = (PROJECT_ROOT / val_path).resolve()
            try:
                annotations = json.loads(annot_path.read_text(encoding="utf-8"))
                label_index = _build_label_index(annotations, str(mode))
                train_classes = _unique_classes_in_dir(train_path, label_index, str(mode))
                val_classes = _unique_classes_in_dir(val_path, label_index, str(mode))
                num_classes_train = len(train_classes)
                num_classes_val = len(val_classes)
            except Exception:
                num_classes_train = 0
                num_classes_val = 0

        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append(
            {
                "f1": f"{stats['f1']:.4f}",
                "fer": f"{stats['fer']:.4f}",
                "probe_mode": probe_mode,
                "mode": mode,
                "species": species,
                "bird": bird,
                "train_seconds": train_seconds,
                "num_classes": int(stats["num_classes"]),
                "num_classes_train": num_classes_train,
                "num_classes_val": num_classes_val,
                "patch_width": int(stats["patch_width"]),
                "frozen_layers": frozen_layers,
                "steps": steps,
                "lr": lr,
                "batch_size": batch_size,
                "class_weighting": class_weighting,
                "run_name": run_dir.name,
                "created_at": created_at,
            }
        )
        records.append(
            {
                "probe_mode": probe_mode,
                "run_name": run_dir.name,
                "mode": stats["mode"],
                "species": species,
                "f1": float(stats["f1"]),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in header})

    # Summary: mean/std per (probe_mode, mode, species) and per (probe_mode, mode) across species.
    summary_rows: List[str] = ["probe_mode,mode,species,n,f1_mean,f1_std"]
    by_group: Dict[Tuple[str, str, str], List[float]] = {}
    for rec in records:
        key = (str(rec["probe_mode"]), str(rec["mode"]), str(rec["species"]))
        by_group.setdefault(key, []).append(float(rec["f1"]))

    for (probe_mode, mode, species), f1s in sorted(by_group.items()):
        arr = np.array(f1s, dtype=np.float64)
        mean = float(arr.mean()) if len(arr) else 0.0
        std = float(arr.std(ddof=0)) if len(arr) else 0.0
        summary_rows.append(f"{probe_mode},{mode},{species},{len(arr)},{mean:.4f},{std:.4f}")

    by_species_mean: Dict[Tuple[str, str], List[float]] = {}
    for (probe_mode, mode, _species), f1s in by_group.items():
        arr = np.array(f1s, dtype=np.float64)
        mean = float(arr.mean()) if len(arr) else 0.0
        by_species_mean.setdefault((probe_mode, mode), []).append(mean)

    for (probe_mode, mode), means in sorted(by_species_mean.items()):
        arr = np.array(means, dtype=np.float64)
        mean = float(arr.mean()) if len(arr) else 0.0
        std = float(arr.std(ddof=0)) if len(arr) else 0.0
        summary_rows.append(f"{probe_mode},{mode},ALL,{len(arr)},{mean:.4f},{std:.4f}")

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.write_text("\n".join(summary_rows) + "\n", encoding="utf-8")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
