#!/usr/bin/env python3
import argparse
from pathlib import Path

from _duration_minmax_table_common import build_duration_min_max_table


def main():
    ap = argparse.ArgumentParser(
        description="Generate min/max duration summary table for syllable classification sweep."
    )
    ap.add_argument(
        "eval_csv",
        nargs="?",
        default="results/duration_classify_sweep/eval_f1.csv",
        help="Path to classify sweep eval_f1.csv (default: results/duration_classify_sweep/eval_f1.csv)",
    )
    ap.add_argument(
        "--out_csv",
        default=None,
        help="Output CSV path (default: <eval_csv_dir>/duration_classify_min_max_table.csv)",
    )
    ap.add_argument("--probe_mode", default="lora", help="Use empty string to disable filtering.")
    ap.add_argument("--species", default="Canary,Zebra_Finch,Bengalese_Finch")
    ap.add_argument("--model_name", default="SongMAE")
    ap.add_argument("--precision", type=int, default=2)
    args = ap.parse_args()

    eval_path = Path(args.eval_csv)
    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else (eval_path if not eval_path.is_dir() else eval_path / "eval_f1.csv").parent
        / "duration_classify_min_max_table.csv"
    )

    build_duration_min_max_table(
        eval_csv=args.eval_csv,
        out_csv=out_csv,
        mode="classify",
        probe_mode=args.probe_mode,
        species_order=[x.strip() for x in args.species.split(",") if x.strip()],
        model_name=args.model_name,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()

