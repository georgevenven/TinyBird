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
        description=(
            "Generate per-species LR sweep tables (averaged across individuals within each species) "
            "from eval_f1.csv for AVES finetuning."
        )
    )
    ap.add_argument(
        "eval_csv",
        nargs="?",
        default="results/aves_lr_sweep/eval_f1.csv",
        help="Path to eval_f1.csv or directory containing it.",
    )
    ap.add_argument("--out_dir", default=None, help="Output directory for generated tables.")
    ap.add_argument("--mode", default="classify")
    ap.add_argument(
        "--probe_mode",
        default="",
        help=(
            "Optional probe_mode filter. Default is empty (disabled) because AVES LR sweeps may be "
            "logged as probe_mode=unknown."
        ),
    )
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

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "lr_sweep_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # agg[species][lr] = {"f1":[...], "fer":[...], "birds":set()}
    agg = defaultdict(lambda: defaultdict(lambda: {"f1": [], "fer": [], "birds": set()}))
    rows_used = 0
    skipped_mismatch = 0

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

            lr = to_float_or_none(row.get("lr", ""))
            f1 = to_float_or_none(row.get("f1", ""))
            fer = to_float_or_none(row.get("fer", ""))
            if lr is None or f1 is None or fer is None:
                continue

            bird = (row.get("bird") or "").strip()
            agg[species][lr]["f1"].append(f1)
            agg[species][lr]["fer"].append(fer)
            if bird:
                agg[species][lr]["birds"].add(bird)
            rows_used += 1

    if rows_used == 0:
        raise SystemExit("No matching rows found after filtering.")

    for species in species_order:
        by_lr = agg.get(species, {})
        if not by_lr:
            continue
        lrs = sorted(by_lr.keys())

        grid_path = out_dir / f"{slug_species(species)}_lr_table.csv"
        with grid_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric"] + [f"lr={lr:g}" for lr in lrs])
            writer.writerow(
                ["F1 (mean +- std)"] + [fmt_mean_std(by_lr[lr]["f1"], args.precision) for lr in lrs]
            )
            writer.writerow(
                ["FER (mean +- std)"] + [fmt_mean_std(by_lr[lr]["fer"], args.precision) for lr in lrs]
            )
            writer.writerow(
                ["N birds"] + [str(len(by_lr[lr]["birds"])) for lr in lrs]
            )
            writer.writerow(
                ["N runs"] + [str(len(by_lr[lr]["f1"])) for lr in lrs]
            )

        ranking_rows = []
        for lr in lrs:
            f1_mean, f1_std = mean_std(by_lr[lr]["f1"])
            fer_mean, fer_std = mean_std(by_lr[lr]["fer"])
            ranking_rows.append(
                {
                    "lr": lr,
                    "n_birds": len(by_lr[lr]["birds"]),
                    "n_runs": len(by_lr[lr]["f1"]),
                    "f1_mean": f1_mean,
                    "f1_std": f1_std,
                    "fer_mean": fer_mean,
                    "fer_std": fer_std,
                }
            )

        ranking_rows.sort(key=lambda x: (-x["f1_mean"], x["fer_mean"], x["lr"]))
        rank_f1_path = out_dir / f"{slug_species(species)}_ranking_by_f1.csv"
        with rank_f1_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank_by_f1",
                    "lr",
                    "n_birds",
                    "n_runs",
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
                        f"{row['lr']:g}",
                        row["n_birds"],
                        row["n_runs"],
                        f"{row['f1_mean']:.{args.precision}f}",
                        f"{row['f1_std']:.{args.precision}f}",
                        f"{row['fer_mean']:.{args.precision}f}",
                        f"{row['fer_std']:.{args.precision}f}",
                    ]
                )

        ranking_rows.sort(key=lambda x: (x["fer_mean"], -x["f1_mean"], x["lr"]))
        rank_fer_path = out_dir / f"{slug_species(species)}_ranking_by_fer.csv"
        with rank_fer_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank_by_fer",
                    "lr",
                    "n_birds",
                    "n_runs",
                    "fer_mean",
                    "fer_std",
                    "f1_mean",
                    "f1_std",
                ]
            )
            for idx, row in enumerate(ranking_rows, start=1):
                writer.writerow(
                    [
                        idx,
                        f"{row['lr']:g}",
                        row["n_birds"],
                        row["n_runs"],
                        f"{row['fer_mean']:.{args.precision}f}",
                        f"{row['fer_std']:.{args.precision}f}",
                        f"{row['f1_mean']:.{args.precision}f}",
                        f"{row['f1_std']:.{args.precision}f}",
                    ]
                )

        print(f"Wrote: {grid_path}")
        print(f"Wrote: {rank_f1_path}")
        print(f"Wrote: {rank_fer_path}")

    print(f"Rows used: {rows_used}")
    print(f"Rows skipped (class mismatch): {skipped_mismatch}")


if __name__ == "__main__":
    main()
