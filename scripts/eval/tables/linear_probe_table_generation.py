#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import pstdev

ALIASES = {
    "canary": "Canary",
    "zebra finch": "Zebra_Finch",
    "zebra_finch": "Zebra_Finch",
    "bengalese finch": "Bengalese_Finch",
    "bengalese_finch": "Bengalese_Finch",
}
DISPLAY = {"Canary": "Canary", "Zebra_Finch": "Zebra Finch", "Bengalese_Finch": "Bengalese Finch"}


def canon_species(raw):
    x = (raw or "").strip()
    if not x:
        return None
    if x in DISPLAY:
        return x
    return ALIASES.get(x.lower().replace("_", " ")) or ALIASES.get(x.lower())


def infer_model(row, forced_model=None):
    if forced_model:
        return forced_model
    pre = (row.get("pretrained_run") or "").strip()
    if pre:
        return Path(pre).name
    run = Path((row.get("run_name") or "").strip()).name
    prefix = "linear_probe_"
    if run.startswith(prefix):
        tail = run[len(prefix) :]
        for species in ("Bengalese_Finch", "Zebra_Finch", "Canary"):
            token = f"_{species}_"
            idx = tail.rfind(token)
            if idx > 0:
                return tail[:idx]
    # AVES-style run names are often just "<Species>_<BirdId>".
    for species in ("Bengalese_Finch", "Zebra_Finch", "Canary"):
        if run.startswith(species + "_"):
            return "AVES"
    return run or "unknown_model"


def fmt(vals, p):
    if not vals:
        return "-"
    mean = sum(vals) / len(vals)
    std = pstdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:.{p}f} +- {std:.{p}f}"


def to_int_or_none(raw):
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description="Generate linear-probe table CSV from eval_f1.csv.")
    ap.add_argument("eval_csv", help="Path to eval_f1.csv or directory containing it")
    ap.add_argument("--out_csv", default=None, help="Output CSV path")
    ap.add_argument("--mode", default="classify")
    ap.add_argument("--probe_mode", default="linear", help="Use empty string to disable")
    ap.add_argument("--species", default="Canary,Zebra_Finch,Bengalese_Finch")
    ap.add_argument("--precision", type=int, default=2)
    ap.add_argument("--model_name", default=None, help="Force one shared model label for all rows.")
    ap.add_argument("--aves", action="store_true", help="Shortcut for --model_name AVES.")
    args = ap.parse_args()

    forced_model = args.model_name or ("AVES" if args.aves else None)

    csv_path = Path(args.eval_csv)
    if csv_path.is_dir():
        csv_path = csv_path / "eval_f1.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing eval_f1.csv: {csv_path}")

    species_order = [canon_species(x.strip()) for x in args.species.split(",")]
    species_order = [x for x in species_order if x]
    if not species_order:
        raise SystemExit("No valid species.")

    agg = defaultdict(lambda: defaultdict(lambda: {"f1": [], "fer": []}))
    model_order, rows_used, rows_skipped_mismatch = [], 0, 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if args.mode and (row.get("mode") or "").strip() != args.mode:
                continue
            if args.probe_mode and (row.get("probe_mode") or "").strip() != args.probe_mode:
                continue
            sp = canon_species(row.get("species", ""))
            if sp not in species_order:
                continue

            nc = to_int_or_none(row.get("num_classes", ""))
            nct = to_int_or_none(row.get("num_classes_train", ""))
            ncv = to_int_or_none(row.get("num_classes_val", ""))
            if nc is None or nct is None or ncv is None or not (nc == nct == ncv):
                rows_skipped_mismatch += 1
                run = (row.get("run_name") or "").strip() or "unknown_run"
                bird = (row.get("bird") or "").strip() or "unknown_bird"
                print(
                    "Skipping row due to class-count mismatch: "
                    f"run={run}, species={sp}, bird={bird}, "
                    f"num_classes={row.get('num_classes', '')}, "
                    f"num_classes_train={row.get('num_classes_train', '')}, "
                    f"num_classes_val={row.get('num_classes_val', '')}"
                )
                continue

            try:
                f1, fer = float(row.get("f1", "")), float(row.get("fer", ""))
            except ValueError:
                continue
            mid = infer_model(row, forced_model=forced_model)
            if mid not in model_order:
                model_order.append(mid)
            agg[mid][sp]["f1"].append(f1)
            agg[mid][sp]["fer"].append(fer)
            rows_used += 1
    if rows_used == 0:
        raise SystemExit("No rows matched filters.")

    out_csv = Path(args.out_csv) if args.out_csv else csv_path.parent / "linear_probe_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = ["Model"] + [
        cell
        for sp in species_order
        for cell in (f"{DISPLAY[sp]} F1", f"{DISPLAY[sp]} FER")
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for mid in model_order:
            row = [mid.replace("_", " ")]
            for sp in species_order:
                row.append(fmt(agg[mid][sp]["f1"], args.precision))
                row.append(fmt(agg[mid][sp]["fer"], args.precision))
            writer.writerow(row)

    print(f"Rows used: {rows_used}")
    print(f"Rows skipped (num_classes mismatch): {rows_skipped_mismatch}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
