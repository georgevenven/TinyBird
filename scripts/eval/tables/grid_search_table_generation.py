#!/usr/bin/env python3
import argparse
import csv
import re
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
DISPLAY = {
    "Canary": "Canary",
    "Zebra_Finch": "Zebra Finch",
    "Bengalese_Finch": "Bengalese Finch",
}


def canon_species(raw):
    text = (raw or "").strip()
    if not text:
        return None
    if text in DISPLAY:
        return text
    return ALIASES.get(text.lower().replace("_", " ")) or ALIASES.get(text.lower())


def to_int_or_none(raw):
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def to_float_or_none(raw):
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_lora_rank(run_name):
    m = re.search(r"_r(\d+)_lr", run_name or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def fmt_mean_std(values, precision):
    if not values:
        return "-"
    mean = sum(values) / len(values)
    std = pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.{precision}f} +- {std:.{precision}f}"


def mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    std = pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def slug_species(species):
    return species.lower().replace(" ", "_")


def main():
    ap = argparse.ArgumentParser(
        description="Generate per-species LoRA-rank x LR grid tables and rankings from eval_f1.csv."
    )
    ap.add_argument("eval_csv", help="Path to eval_f1.csv or directory containing it.")
    ap.add_argument("--out_dir", default=None, help="Output directory for generated tables.")
    ap.add_argument("--mode", default="classify")
    ap.add_argument("--probe_mode", default="lora", help="Use empty string to disable.")
    ap.add_argument("--species", default="Canary,Zebra_Finch,Bengalese_Finch")
    ap.add_argument("--precision", type=int, default=2)
    args = ap.parse_args()

    csv_path = Path(args.eval_csv)
    if csv_path.is_dir():
        csv_path = csv_path / "eval_f1.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing eval_f1.csv: {csv_path}")

    species_order = [canon_species(x.strip()) for x in args.species.split(",")]
    species_order = [x for x in species_order if x]
    if not species_order:
        raise SystemExit("No valid species.")
    species_set = set(species_order)

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "grid_search_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # agg[species][rank][lr] = {"f1":[...], "fer":[...]}
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"f1": [], "fer": []})))
    rows_used = 0
    skipped_mismatch = 0
    skipped_missing_rank = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if args.mode and (row.get("mode") or "").strip() != args.mode:
                continue
            if args.probe_mode and (row.get("probe_mode") or "").strip() != args.probe_mode:
                continue

            species = canon_species(row.get("species", ""))
            if species not in species_set:
                continue

            nc = to_int_or_none(row.get("num_classes", ""))
            nct = to_int_or_none(row.get("num_classes_train", ""))
            ncv = to_int_or_none(row.get("num_classes_val", ""))
            if nc is None or nct is None or ncv is None or not (nc == nct == ncv):
                skipped_mismatch += 1
                print(
                    "Skipping row due to class-count mismatch: "
                    f"run={row.get('run_name','')}, species={species}, bird={row.get('bird','')}, "
                    f"num_classes={row.get('num_classes','')}, "
                    f"num_classes_train={row.get('num_classes_train','')}, "
                    f"num_classes_val={row.get('num_classes_val','')}"
                )
                continue

            rank = parse_lora_rank(row.get("run_name", ""))
            if rank is None:
                skipped_missing_rank += 1
                print(
                    "Skipping row with missing LoRA rank in run_name: "
                    f"run={row.get('run_name','')}, species={species}, bird={row.get('bird','')}"
                )
                continue

            lr = to_float_or_none(row.get("lr", ""))
            f1 = to_float_or_none(row.get("f1", ""))
            fer = to_float_or_none(row.get("fer", ""))
            if lr is None or f1 is None or fer is None:
                continue

            agg[species][rank][lr]["f1"].append(f1)
            agg[species][rank][lr]["fer"].append(fer)
            rows_used += 1

    if rows_used == 0:
        raise SystemExit("No matching rows found after filtering.")

    for species in species_order:
        by_rank = agg.get(species, {})
        if not by_rank:
            continue

        ranks = sorted(by_rank.keys())
        lrs = sorted({lr for rank in by_rank for lr in by_rank[rank]})

        # Grid table (rows = LoRA rank, cols = LR), values = mean+-std F1
        grid_path = out_dir / f"{slug_species(species)}_rank_lr_table.csv"
        with grid_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            header = ["lora_rank"] + [f"lr={lr:g}" for lr in lrs]
            writer.writerow(header)
            for rank in ranks:
                row = [rank]
                for lr in lrs:
                    vals = by_rank[rank].get(lr, {}).get("f1", [])
                    row.append(fmt_mean_std(vals, args.precision))
                writer.writerow(row)

        # Ranking table by mean F1 (desc), tie-break on mean FER (asc)
        ranking_rows = []
        for rank in ranks:
            for lr in lrs:
                stats = by_rank[rank].get(lr)
                if not stats:
                    continue
                f1_vals = stats["f1"]
                fer_vals = stats["fer"]
                if not f1_vals:
                    continue
                f1_mean, f1_std = mean_std(f1_vals)
                fer_mean, fer_std = mean_std(fer_vals)
                ranking_rows.append(
                    {
                        "lora_rank": rank,
                        "lr": lr,
                        "n": len(f1_vals),
                        "f1_mean": f1_mean,
                        "f1_std": f1_std,
                        "fer_mean": fer_mean,
                        "fer_std": fer_std,
                    }
                )

        ranking_rows.sort(key=lambda x: (-x["f1_mean"], x["fer_mean"], x["lora_rank"], x["lr"]))
        ranking_path = out_dir / f"{slug_species(species)}_ranking_by_f1.csv"
        with ranking_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank_by_f1",
                    "lora_rank",
                    "lr",
                    "n",
                    "f1_mean",
                    "f1_std",
                    "fer_mean",
                    "fer_std",
                ]
            )
            for idx, row in enumerate(ranking_rows, start=1):
                writer.writerow(
                    [
                        idx,
                        row["lora_rank"],
                        f"{row['lr']:g}",
                        row["n"],
                        f"{row['f1_mean']:.{args.precision}f}",
                        f"{row['f1_std']:.{args.precision}f}",
                        f"{row['fer_mean']:.{args.precision}f}",
                        f"{row['fer_std']:.{args.precision}f}",
                    ]
                )

        print(f"Wrote: {grid_path}")
        print(f"Wrote: {ranking_path}")

    print(f"Rows used: {rows_used}")
    print(f"Rows skipped (class mismatch): {skipped_mismatch}")
    print(f"Rows skipped (missing rank in run_name): {skipped_missing_rank}")


if __name__ == "__main__":
    main()
