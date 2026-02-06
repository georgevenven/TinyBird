#!/usr/bin/env python3
import csv
from pathlib import Path

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


def fmt_score(f1_vals, fer_vals, precision):
    if not f1_vals or not fer_vals:
        return "-"
    f1_mean = sum(f1_vals) / len(f1_vals)
    fer_mean = sum(fer_vals) / len(fer_vals)
    return f"{f1_mean:.{precision}f} / {fer_mean:.{precision}f}"


def _norm_seconds(text):
    v = (text or "").strip()
    if not v:
        return None
    if v.upper() == "MAX":
        return "MAX"
    try:
        return float(v)
    except ValueError:
        return None


def _seconds_label(seconds_value):
    if seconds_value == "MAX":
        return "MAX"
    if float(seconds_value).is_integer():
        return f"{int(seconds_value)}s"
    return f"{seconds_value:g}s"


def _select_min_max_seconds(seconds_values):
    numeric = sorted({x for x in seconds_values if isinstance(x, float)})
    has_max = "MAX" in seconds_values

    # MAX is a sentinel that represents "all available data", which is
    # logically larger than any explicit numeric duration.
    if numeric and has_max:
        return numeric[0], "MAX"
    if numeric:
        return numeric[0], numeric[-1]
    if has_max:
        return "MAX", "MAX"
    return None, None


def build_duration_min_max_table(
    *,
    eval_csv,
    out_csv,
    mode,
    probe_mode,
    species_order,
    model_name,
    precision,
):
    eval_path = Path(eval_csv)
    if eval_path.is_dir():
        eval_path = eval_path / "eval_f1.csv"
    if not eval_path.exists():
        raise SystemExit(f"Missing eval_f1.csv: {eval_path}")

    canonical_species = []
    for sp in species_order:
        c = canon_species(sp)
        if c and c not in canonical_species:
            canonical_species.append(c)
    if not canonical_species:
        raise SystemExit("No valid species.")
    species_set = set(canonical_species)

    # data[species][seconds] = {"f1":[...], "fer":[...]}
    data = {
        sp: {}
        for sp in canonical_species
    }
    rows_used = 0
    rows_skipped_mismatch = 0

    with eval_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if mode and (row.get("mode") or "").strip() != mode:
                continue
            if probe_mode and (row.get("probe_mode") or "").strip() != probe_mode:
                continue

            sp = canon_species(row.get("species", ""))
            if sp not in species_set:
                continue

            nc = to_int_or_none(row.get("num_classes", ""))
            nct = to_int_or_none(row.get("num_classes_train", ""))
            ncv = to_int_or_none(row.get("num_classes_val", ""))
            if nc is None or nct is None or ncv is None or not (nc == nct == ncv):
                rows_skipped_mismatch += 1
                print(
                    "Skipping row due to class-count mismatch: "
                    f"run={row.get('run_name','')}, species={sp}, bird={row.get('bird','')}, "
                    f"num_classes={row.get('num_classes','')}, "
                    f"num_classes_train={row.get('num_classes_train','')}, "
                    f"num_classes_val={row.get('num_classes_val','')}"
                )
                continue

            secs = _norm_seconds(row.get("train_seconds", ""))
            f1 = to_float_or_none(row.get("f1", ""))
            fer = to_float_or_none(row.get("fer", ""))
            if secs is None or f1 is None or fer is None:
                continue

            bucket = data[sp].setdefault(secs, {"f1": [], "fer": []})
            bucket["f1"].append(f1)
            bucket["fer"].append(fer)
            rows_used += 1

    if rows_used == 0:
        raise SystemExit("No matching rows found after filtering.")

    # Build one-row SongMAE table with per-species min/max duration columns.
    header = ["Model"]
    out_row = [model_name]
    for sp in canonical_species:
        sec_keys = list(data[sp].keys())
        min_sec, max_sec = _select_min_max_seconds(sec_keys)
        if min_sec is None:
            header.extend([f"{DISPLAY[sp]} Min", f"{DISPLAY[sp]} Max"])
            out_row.extend(["-", "-"])
            continue

        min_label = f"{DISPLAY[sp]} {_seconds_label(min_sec)} (Min)"
        max_label = f"{DISPLAY[sp]} {_seconds_label(max_sec)} (Max)"
        header.extend([min_label, max_label])

        min_stats = data[sp].get(min_sec, {"f1": [], "fer": []})
        max_stats = data[sp].get(max_sec, {"f1": [], "fer": []})
        out_row.append(fmt_score(min_stats["f1"], min_stats["fer"], precision))
        out_row.append(fmt_score(max_stats["f1"], max_stats["fer"], precision))

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(out_row)

    print(f"Rows used: {rows_used}")
    print(f"Rows skipped (num_classes mismatch): {rows_skipped_mismatch}")
    print(f"Wrote: {out_path}")
