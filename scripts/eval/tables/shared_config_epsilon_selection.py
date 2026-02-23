#!/usr/bin/env python3
"""
Select a single grid-search (LoRA rank, LR) config that works across species via an epsilon criterion.

Using eval_f1.csv:
  - Drop *exact* duplicate rows (all columns identical).
  - If multiple runs exist for the same (species, bird, rank, lr), average within-bird first.
  - Then average equally across birds to get per-species mean F1/Fer for each config.
  - For each species s, compute best_s = max_c F1_s(c).
  - For each config c, compute g(c) = max_s (best_s - F1_s(c)).
    Config c is within epsilon for all species iff g(c) <= epsilon.

Outputs:
  - Prints best per-species configs and the minimax config (smallest g).
  - Optionally prints feasible sets for requested epsilons.
  - Writes a ranked CSV with g(c), per-species means, and gaps.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple


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


def canon_species(raw: str) -> Optional[str]:
    text = (raw or "").strip()
    if not text:
        return None
    if text in DISPLAY:
        return text
    return ALIASES.get(text.lower().replace("_", " ")) or ALIASES.get(text.lower())


def parse_lora_rank(run_name: str) -> Optional[int]:
    m = re.search(r"_r(\d+)_lr", run_name or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def to_float_or_none(raw: str) -> Optional[float]:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def resolve_eval_csv(path: Path) -> Path:
    p = path
    if p.is_dir():
        p = p / "eval_f1.csv"
    if not p.exists():
        raise SystemExit(f"Missing eval_f1.csv: {p}")
    return p


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for item in parse_csv_list(raw):
        try:
            out.append(float(item))
        except ValueError as exc:
            raise SystemExit(f"Invalid float in list: {item}") from exc
    return out


def fmt_lr(lr: float) -> str:
    # Prefer canonical tags used in paper text for common values.
    for s in ("5e-3", "1e-3", "5e-4", "1e-4"):
        if abs(lr - float(s)) < 1e-12:
            return s
    return f"{lr:g}"


@dataclass(frozen=True, order=True)
class Config:
    rank: int
    lr: float

    def label(self) -> str:
        return f"(rank {self.rank}, LR {fmt_lr(self.lr)})"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select a shared (rank, lr) config within epsilon of each species' best."
    )
    ap.add_argument("eval_csv", help="Path to eval_f1.csv or directory containing it.")
    ap.add_argument("--out_csv", default=None, help="Output CSV path.")
    ap.add_argument("--out_dir", default=None, help="Output directory (ignored if --out_csv set).")
    ap.add_argument("--mode", default="classify")
    ap.add_argument("--probe_mode", default="lora", help="Use empty string to disable.")
    ap.add_argument(
        "--species",
        default="",
        help="Optional comma-separated species filter (canonical names). Default: all found.",
    )
    ap.add_argument(
        "--epsilons",
        default="",
        help="Optional comma-separated eps values to print feasible sets for (e.g. 2.0,2.5,3.5).",
    )
    ap.add_argument("--top", type=int, default=24, help="How many ranked configs to print.")
    ap.add_argument("--precision", type=int, default=2)
    args = ap.parse_args()

    csv_path = resolve_eval_csv(Path(args.eval_csv))
    species_filter = set(parse_csv_list(args.species))
    eps_list = parse_float_list(args.epsilons) if args.epsilons else []

    # Load + drop exact duplicates.
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"No header found in {csv_path}")
        fieldnames = list(reader.fieldnames)
        seen = set()
        dup_count = 0
        for row in reader:
            key = tuple((row.get(name) or "") for name in fieldnames)
            if key in seen:
                dup_count += 1
                continue
            seen.add(key)
            rows.append(dict(row))

    if not rows:
        raise SystemExit("No rows loaded.")

    # Group run rows by (species, bird, rank, lr) so we can average within bird first.
    # bird_cfg[(sp, bird, cfg)] -> lists of f1/fer across replicate runs
    bird_cfg: Dict[Tuple[str, str, Config], Dict[str, List[float]]] = defaultdict(
        lambda: {"f1": [], "fer": []}
    )
    rows_used = 0
    skipped = 0
    for row in rows:
        if args.mode and (row.get("mode") or "").strip() != args.mode:
            continue
        if args.probe_mode and (row.get("probe_mode") or "").strip() != args.probe_mode:
            continue

        sp = canon_species(row.get("species", ""))
        if not sp or (species_filter and sp not in species_filter):
            continue

        bird = (row.get("bird") or row.get("bird_id") or "").strip()
        rank = parse_lora_rank(row.get("run_name", ""))
        lr = to_float_or_none(row.get("lr", ""))
        f1 = to_float_or_none(row.get("f1", ""))
        fer = to_float_or_none(row.get("fer", ""))
        if not bird or rank is None or lr is None or f1 is None or fer is None:
            skipped += 1
            continue

        cfg = Config(rank=rank, lr=lr)
        bird_cfg[(sp, bird, cfg)]["f1"].append(f1)
        bird_cfg[(sp, bird, cfg)]["fer"].append(fer)
        rows_used += 1

    if rows_used == 0:
        raise SystemExit("No rows matched filters.")

    # Average within bird.
    bird_cfg_mean: Dict[Tuple[str, str, Config], Dict[str, float]] = {}
    multi_run_groups = 0
    for key, agg in bird_cfg.items():
        if len(agg["f1"]) > 1:
            multi_run_groups += 1
        bird_cfg_mean[key] = {"f1": mean(agg["f1"]), "fer": mean(agg["fer"]), "n_runs": float(len(agg["f1"]))}

    # Average equally across birds within each species.
    # species_cfg[(sp, cfg)] -> lists of per-bird means
    species_cfg: Dict[Tuple[str, Config], Dict[str, List[float]]] = defaultdict(
        lambda: {"f1": [], "fer": [], "n_runs": [], "birds": []}
    )
    for (sp, bird, cfg), vals in bird_cfg_mean.items():
        species_cfg[(sp, cfg)]["f1"].append(vals["f1"])
        species_cfg[(sp, cfg)]["fer"].append(vals["fer"])
        species_cfg[(sp, cfg)]["n_runs"].append(vals["n_runs"])
        species_cfg[(sp, cfg)]["birds"].append(bird)

    species_list = sorted({sp for sp, _ in species_cfg.keys()})
    if not species_list:
        raise SystemExit("No species after filtering.")

    # Best config per species.
    best_by_species: Dict[str, Tuple[Config, float, int]] = {}
    for sp in species_list:
        candidates: List[Tuple[Config, float, int]] = []
        for (sp2, cfg), agg in species_cfg.items():
            if sp2 != sp:
                continue
            n_birds = len(set(agg["birds"]))
            f1_mean = mean(agg["f1"])
            candidates.append((cfg, f1_mean, n_birds))
        candidates.sort(key=lambda x: (-x[1], x[0].rank, x[0].lr))
        best_by_species[sp] = candidates[0]

    # For each config, compute g(c) across species (requires config present for all selected species).
    all_cfgs = sorted({cfg for _, cfg in species_cfg.keys()})
    rows_out: List[Dict[str, object]] = []
    for cfg in all_cfgs:
        per_sp = {}
        complete = True
        for sp in species_list:
            agg = species_cfg.get((sp, cfg))
            if not agg:
                complete = False
                break
            per_sp[sp] = {
                "f1_mean": mean(agg["f1"]),
                "fer_mean": mean(agg["fer"]),
                "n_birds": float(len(set(agg["birds"]))),
                "n_runs": float(sum(agg["n_runs"])),
            }
        if not complete:
            continue
        gaps = {sp: (best_by_species[sp][1] - per_sp[sp]["f1_mean"]) for sp in species_list}
        g = max(gaps.values())
        mean_over_species = mean([per_sp[sp]["f1_mean"] for sp in species_list])
        rows_out.append(
            {
                "cfg": cfg,
                "g": g,
                "mean_over_species": mean_over_species,
                "per_sp": per_sp,
                "gaps": gaps,
            }
        )

    if not rows_out:
        raise SystemExit("No configs had data for all selected species.")

    rows_out.sort(key=lambda r: (r["g"], -r["mean_over_species"], r["cfg"].rank, r["cfg"].lr))

    print(f"Exact-duplicate check: {len(rows) + dup_count} rows → {len(rows)} rows after drop_duplicates() ({dup_count} exact repeats).")
    if multi_run_groups:
        print(
            f"There are {multi_run_groups} (species,bird,rank,lr) combinations with multiple runs; "
            "averaging within bird first, then averaging across birds within species."
        )
    if skipped:
        print(f"Skipped {skipped} rows with missing/invalid fields after filters.")

    print()
    print("1) Best mean F1 per species (mean across individuals within species)")
    for sp in species_list:
        cfg, best_f1, n_birds = best_by_species[sp]
        print(
            f"{DISPLAY.get(sp, sp)} (n={n_birds} birds): best = {best_f1:.{args.precision}f} at {cfg.label()}"
        )

    print()
    print("2) “Within ε for all species” recalculation")
    best_row = rows_out[0]
    best_cfg: Config = best_row["cfg"]
    g_star: float = best_row["g"]
    parts = []
    for sp in species_list:
        f1m = best_row["per_sp"][sp]["f1_mean"]
        gap = best_row["gaps"][sp]
        parts.append(f"{DISPLAY.get(sp, sp)} {f1m:.{args.precision}f} (gap {gap:.{args.precision}f})")
    print(f"{best_cfg.label()}: g={g_star:.{args.precision}f}")
    print("Mean F1 by species: " + ", ".join(parts))

    top_n = max(0, int(args.top))
    if top_n:
        print()
        print(f"Ranked configs by g(c) (top {min(top_n, len(rows_out))}):")
        for idx, row in enumerate(rows_out[:top_n], start=1):
            cfg: Config = row["cfg"]
            g = row["g"]
            mean_over = row["mean_over_species"]
            per = row["per_sp"]
            per_str = ", ".join(
                f"{DISPLAY.get(sp, sp)}={per[sp]['f1_mean']:.{args.precision}f}" for sp in species_list
            )
            print(
                f"{idx:>2d}. {cfg.label()}  g={g:.{args.precision}f}  "
                f"mean_over_species={mean_over:.{args.precision}f}  {per_str}"
            )

    if eps_list:
        print()
        for eps in eps_list:
            feasible = [r for r in rows_out if r["g"] <= eps + 1e-12]
            cfgs = [(r["cfg"].rank, fmt_lr(r["cfg"].lr)) for r in feasible]
            cfgs_str = ", ".join(f"({rank}, {lr})" for rank, lr in cfgs) if cfgs else "(none)"
            print(f"ε = {eps:g}: {cfgs_str}")

    # Write CSV output.
    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "grid_search_tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "shared_config_epsilon_selection.csv"

    header = ["rank_by_g", "lora_rank", "lr", "g", "mean_f1_over_species"]
    for sp in species_list:
        slug = sp.lower()
        header.extend(
            [
                f"{slug}_f1_mean",
                f"{slug}_gap_to_best",
                f"{slug}_fer_mean",
                f"{slug}_n_birds",
                f"{slug}_n_runs",
            ]
        )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for idx, row in enumerate(rows_out, start=1):
            cfg: Config = row["cfg"]
            record: Dict[str, object] = {
                "rank_by_g": idx,
                "lora_rank": cfg.rank,
                "lr": f"{cfg.lr:g}",
                "g": f"{row['g']:.{args.precision}f}",
                "mean_f1_over_species": f"{row['mean_over_species']:.{args.precision}f}",
            }
            for sp in species_list:
                slug = sp.lower()
                record[f"{slug}_f1_mean"] = f"{row['per_sp'][sp]['f1_mean']:.{args.precision}f}"
                record[f"{slug}_gap_to_best"] = f"{row['gaps'][sp]:.{args.precision}f}"
                record[f"{slug}_fer_mean"] = f"{row['per_sp'][sp]['fer_mean']:.{args.precision}f}"
                record[f"{slug}_n_birds"] = int(row["per_sp"][sp]["n_birds"])
                record[f"{slug}_n_runs"] = int(row["per_sp"][sp]["n_runs"])
            writer.writerow(record)

    print()
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()

