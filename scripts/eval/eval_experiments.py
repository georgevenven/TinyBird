#!/usr/bin/env python3

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "files" / "eval_experiments.json"
EVAL_SCRIPT = ROOT / "scripts" / "eval" / "eval_embedding.py"


def ensure_abs(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def load_config(path):
    with Path(path).expanduser().open("r") as handle:
        return json.load(handle)


def slugify(value):
    text = str(value)
    for sep in ("/", "\\"):
        text = text.replace(sep, "_")
    return text.replace(" ", "_")


def merge_eval_args(*levels):
    merged = {}
    for level in levels:
        if not level or not isinstance(level, dict):
            continue
        merged.update(level.get("eval_args", {}))
    return merged


def _normalize_run(entry, defaults):
    if isinstance(entry, str):
        run_dir = ensure_abs(entry)
        name = Path(entry).name
        checkpoint = None
        data = {}
    else:
        name = entry.get("name")
        if entry.get("run_dir"):
            run_dir = ensure_abs(entry["run_dir"])
        else:
            base = defaults.get("run_base", ROOT / "runs")
            if name is None:
                raise ValueError("Run entry missing 'name' when 'run_dir' not provided.")
            run_dir = ensure_abs(base / name)
        if name is None:
            name = Path(run_dir).name
        checkpoint = entry.get("checkpoint")
        data = entry
    return {
        "name": name,
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "extra": data,
    }


def _normalize_dataset(entry):
    spec_dir = ensure_abs(entry["spec_dir"])
    annotations = ensure_abs(entry["annotations"])
    label = entry.get("name") or spec_dir.name
    bird_entries = entry.get("birds")
    if bird_entries is None:
        bird_entries = entry.get("bird_ids", [])
    return {
        "spec_dir": spec_dir,
        "annotations": annotations,
        "birds": bird_entries,
        "label": label,
        "extra": entry,
    }


def _normalize_bird(entry, base_spec_dir):
    if isinstance(entry, str):
        bird_id = entry
        spec_override = None
        data = {}
    else:
        bird_id = entry["bird_id"] if "bird_id" in entry else entry["name"]
        spec_override = entry.get("spec_dir")
        data = entry
    if spec_override:
        spec_dir = ensure_abs(spec_override)
    else:
        candidate = base_spec_dir / bird_id
        spec_dir = candidate if candidate.is_dir() else base_spec_dir
    return {
        "bird_id": bird_id,
        "spec_dir": spec_dir,
        "extra": data,
    }


def iter_experiments(config):
    defaults = {
        "run_base": ensure_abs(Path(config.get("run_base", "runs"))),
    }
    runs = [_normalize_run(entry, defaults) for entry in config.get("run_dirs", [])]
    datasets = [_normalize_dataset(entry) for entry in config.get("datasets", [])]

    for run in runs:
        for dataset in datasets:
            for bird_entry in dataset["birds"]:
                bird = _normalize_bird(bird_entry, dataset["spec_dir"])
                yield {
                    "run_name": run["name"],
                    "run_dir": run["run_dir"],
                    "checkpoint": run["checkpoint"],
                    "spec_dir": bird["spec_dir"],
                    "json_path": dataset["annotations"],
                    "bird_id": bird["bird_id"],
                    "extra_args": merge_eval_args(config, run["extra"], dataset["extra"], bird["extra"]),
                    "dataset_label": dataset["label"],
                }


def build_command(exp, out_dir):
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--results_dir",
        str(out_dir),
        "--spec_dir",
        str(exp["spec_dir"]),
        "--run_dir",
        str(exp["run_dir"]),
    ]
    if exp["checkpoint"]:
        cmd.extend(["--checkpoint", str(exp["checkpoint"])])
    if exp["json_path"]:
        cmd.extend(["--json_path", str(exp["json_path"])])
    if exp["bird_id"]:
        cmd.extend(["--bird", str(exp["bird_id"])])
    for key, value in exp["extra_args"].items():
        if key == "pre_encoder":
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            cmd.extend([flag, str(value)])
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Batch TinyBird embedding evaluations.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to eval config JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    config = load_config(args.config)
    results_root = ensure_abs(config.get("results_root", "results/eval_experiments"))
    results_root.mkdir(parents=True, exist_ok=True)

    experiments = list(iter_experiments(config))
    if not experiments:
        print("No experiments defined in config.")
        return

    for exp in experiments:
        out_dir = (
            results_root
            / slugify(exp["run_name"])
            / slugify(exp["dataset_label"])
            / slugify(exp["bird_id"])
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(exp, out_dir)
        if args.dry_run:
            print(" ".join(shlex.quote(part) for part in command))
            continue
        print(f"[eval] {exp['run_name']} / {exp['dataset_label']} / {exp['bird_id']} -> {out_dir}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
