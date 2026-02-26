#!/usr/bin/env python3
import argparse
import csv
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    key: str
    label: str
    color: str


MODELS = (
    ModelConfig("songmae", "SongMAE", "#1f77b4"),  # blue
    ModelConfig("aves", "AVES", "#d62728"),  # red
)

PREFERRED_SPECIES_ORDER = ["Bengalese_Finch", "Zebra_Finch", "Canary"]
SPECIES_DISPLAY = {
    "Bengalese_Finch": "Bengalese Finch",
    "Zebra_Finch": "Zebra Finch",
    "Canary": "Canary",
}
METRIC_LABELS = {"f1": "F1 Score (%)", "fer": "Frame Error Rate (%)"}
METRIC_TITLES = {"f1": "F1", "fer": "FER"}


def _resolve_results_csv(path: str) -> str:
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    candidates = [
        os.path.join(path, "eval_f1.csv"),
        os.path.join(path, "eval_f1_combined.csv"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"No eval_f1 CSV found in {path}. Expected one of: {', '.join(candidates)}"
    )


def _normalize_train_seconds(train_seconds_raw: str) -> tuple[str, tuple[int, float]] | None:
    token = str(train_seconds_raw).strip()
    if not token:
        return None
    if token.upper() == "MAX":
        return ("MAX", (1, float("inf")))
    try:
        numeric = float(token)
    except (TypeError, ValueError):
        return None
    return (f"{numeric:g}", (0, numeric))


def _parse_float(raw: str) -> Optional[float]:
    token = str(raw).strip()
    if not token:
        return None
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def _load_rows(
    csv_path: str,
    *,
    mode: str,
    probe_mode: Optional[str],
) -> list[dict]:
    data: list[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if mode and row.get("mode") != mode:
                continue
            if probe_mode and row.get("probe_mode") != probe_mode:
                continue

            species = (row.get("species") or "").strip()
            bird = (row.get("bird") or "").strip() or "unknown"
            ts_parsed = _normalize_train_seconds(row.get("train_seconds") or "")
            if not species or ts_parsed is None:
                continue
            train_seconds, sort_key = ts_parsed

            f1_metric = _parse_float(row.get("f1") or "")
            fer_metric = _parse_float(row.get("fer") or "")
            if f1_metric is None and fer_metric is None:
                continue

            data.append(
                {
                    "species": species,
                    "bird": bird,
                    "train_seconds": train_seconds,
                    "train_seconds_sort_key": sort_key,
                    "f1": f1_metric,
                    "fer": fer_metric,
                }
            )
    return data


def _species_order(songmae_rows: list[dict], aves_rows: list[dict]) -> list[str]:
    species_set = {row["species"] for row in songmae_rows}
    species_set.update(row["species"] for row in aves_rows)
    ordered = [sp for sp in PREFERRED_SPECIES_ORDER if sp in species_set]
    ordered.extend(sorted(species_set - set(ordered)))
    return ordered


def _build_species_metric_series(rows: list[dict], metric_name: str):
    birds = sorted({row["bird"] for row in rows})
    x_sort_keys: dict[str, tuple[int, float]] = {}
    for row in rows:
        x_sort_keys[row["train_seconds"]] = row["train_seconds_sort_key"]
    x_levels = sorted(x_sort_keys.keys(), key=lambda ts: x_sort_keys[ts])
    x_index = {ts: i for i, ts in enumerate(x_levels)}

    bird_series: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        metric_value = row[metric_name]
        if metric_value is None:
            continue
        bird_series.setdefault(row["bird"], {}).setdefault(row["train_seconds"], []).append(metric_value)

    if not bird_series:
        return None

    avg_by_ts: dict[str, list[float]] = {}
    for by_ts in bird_series.values():
        for ts, vals in by_ts.items():
            avg_by_ts.setdefault(ts, []).append(float(np.mean(vals)))
    avg_xs = sorted(avg_by_ts.keys(), key=lambda ts: x_sort_keys[ts])
    avg_ys = [float(np.mean(avg_by_ts[x])) for x in avg_xs]
    avg_positions = [x_index[x] for x in avg_xs]

    return {
        "birds": birds,
        "x_levels": x_levels,
        "x_sort_keys": x_sort_keys,
        "x_index": x_index,
        "bird_series": bird_series,
        "avg_ys": avg_ys,
        "avg_positions": avg_positions,
    }


def _draw_species_metric_ax(
    ax,
    *,
    metric_name: str,
    model_label: str,
    model_color: str,
    series,
    show_ylabel: bool,
    show_xlabel: bool,
):
    for bird in series["birds"]:
        by_ts = series["bird_series"].get(bird, {})
        if not by_ts:
            continue
        xs = sorted(by_ts.keys(), key=lambda ts: series["x_sort_keys"][ts])
        ys = [float(np.mean(by_ts[x])) for x in xs]
        x_positions = [series["x_index"][x] for x in xs]
        ax.plot(
            x_positions,
            ys,
            marker="o",
            markersize=3.8,
            linewidth=0.95,
            alpha=0.30,
            color=model_color,
        )

    ax.plot(
        series["avg_positions"],
        series["avg_ys"],
        marker="o",
        markersize=4.8,
        linewidth=2.2,
        color=model_color,
        alpha=0.95,
    )

    ax.set_title(f"{METRIC_TITLES[metric_name]} ({model_label})", fontsize=11.5, fontweight="bold")
    if show_xlabel:
        ax.set_xlabel("# Training Seconds", fontsize=10, fontweight="bold")
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(METRIC_LABELS[metric_name], fontsize=10, fontweight="bold")
        # Keep row y-axis labels perfectly aligned across rows.
        ax.yaxis.set_label_coords(-0.18, 0.5)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.grid(True, alpha=0.22)
    ax.set_xlim(-0.25, max(0.25, len(series["x_levels"]) - 0.75))
    ax.set_xticks(list(range(len(series["x_levels"]))))
    ax.set_xticklabels(series["x_levels"])
    ax.xaxis.set_major_locator(ticker.FixedLocator(list(range(len(series["x_levels"])))))
    ax.tick_params(axis="both", labelsize=10.5, width=1.0)

    if metric_name == "f1":
        ax.set_ylim(40.0, 100.0)
    else:
        ax.set_ylim(0.0, 25.0)

    for side in ("top", "bottom", "left", "right"):
        spine = ax.spines[side]
        if spine.get_visible():
            spine.set_linewidth(1.0)
            spine.set_color("#404040")


def _save_plot_with_pdf(fig, png_path: str) -> tuple[str, str]:
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    pdf_path = os.path.splitext(png_path)[0] + ".pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    return png_path, pdf_path


def _make_slug(text: str) -> str:
    import re

    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "all"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a joint comparison figure with the original time-vs-score style, "
            "using model color (SongMAE blue, AVES red). Layout: 2 rows x 6 columns by default."
        )
    )
    parser.add_argument(
        "songmae_results",
        type=str,
        help="SongMAE results directory or direct CSV path.",
    )
    parser.add_argument(
        "aves_results",
        type=str,
        help="AVES results directory or direct CSV path.",
    )
    parser.add_argument("--mode", type=str, default="classify", help="Mode filter (default: classify).")
    parser.add_argument(
        "--probe_mode",
        type=str,
        default=None,
        help="Optional probe_mode filter (default: none).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <songmae_results_dir>/plots_compare_models).",
    )
    args = parser.parse_args()

    songmae_csv = _resolve_results_csv(args.songmae_results)
    aves_csv = _resolve_results_csv(args.aves_results)

    songmae_rows = _load_rows(songmae_csv, mode=args.mode, probe_mode=args.probe_mode)
    aves_rows = _load_rows(aves_csv, mode=args.mode, probe_mode=args.probe_mode)
    if not songmae_rows:
        raise SystemExit(f"No matching rows for SongMAE in {songmae_csv}")
    if not aves_rows:
        raise SystemExit(f"No matching rows for AVES in {aves_csv}")

    species_order = _species_order(songmae_rows, aves_rows)
    if not species_order:
        raise SystemExit("No species found after filtering.")

    # Alternating by species: [BF SongMAE, BF AVES, ZF SongMAE, ZF AVES, ...]
    model_order = list(MODELS)
    n_rows = 2
    n_cols = len(species_order) * len(model_order)

    # Use compact panel sizing so all 12 panels fit in one figure.
    per_panel_w = 2.48
    per_panel_h = 3.28
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(per_panel_w * n_cols, per_panel_h * n_rows),
        dpi=300,
        squeeze=False,
    )

    model_rows = {
        "songmae": songmae_rows,
        "aves": aves_rows,
    }
    row_metrics = ("f1", "fer")

    for row_idx, metric_name in enumerate(row_metrics):
        for species_idx, species in enumerate(species_order):
            for model_idx, model in enumerate(model_order):
                col_idx = species_idx * len(model_order) + model_idx
                ax = axes[row_idx][col_idx]
                rows = [row for row in model_rows[model.key] if row["species"] == species]
                series = _build_species_metric_series(rows, metric_name)
                if series is None:
                    ax.set_visible(False)
                    continue
                _draw_species_metric_ax(
                    ax,
                    metric_name=metric_name,
                    model_label=model.label,
                    model_color=model.color,
                    series=series,
                    show_ylabel=(col_idx == 0),
                    show_xlabel=(row_idx == n_rows - 1),
                )

    fig.subplots_adjust(left=0.04, right=0.995, bottom=0.13, top=0.72, wspace=0.15, hspace=0.24)

    # Reduce spacing within each species pair (SongMAE/AVES) while preserving
    # the larger gaps between species groups.
    if len(model_order) == 2 and len(species_order) > 0:
        gap = axes[0][1].get_position().x0 - axes[0][0].get_position().x1
        half_gap = gap * 0.5
        for species_idx in range(len(species_order)):
            for model_idx in range(len(model_order)):
                col_idx = species_idx * len(model_order) + model_idx
                # Model 0: shift left by species_idx * half_gap
                # Model 1: shift left by (species_idx + 1) * half_gap
                shift = -half_gap * (species_idx + model_idx)
                for row_idx in range(n_rows):
                    ax = axes[row_idx][col_idx]
                    pos = ax.get_position()
                    ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])

    # Species headers centered above each adjacent SongMAE/AVES pair.
    for species_idx, species in enumerate(species_order):
        left_col = species_idx * len(model_order)
        right_col = left_col + len(model_order) - 1
        left_pos = axes[0][left_col].get_position()
        right_pos = axes[0][right_col].get_position()
        x_center = (left_pos.x0 + right_pos.x1) * 0.5
        y_header = max(left_pos.y1, right_pos.y1) + 0.07
        species_title = SPECIES_DISPLAY.get(species, species.replace("_", " "))
        fig.text(
            x_center,
            y_header,
            species_title,
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    if args.output_dir:
        output_dir = args.output_dir
    else:
        base_dir = args.songmae_results if os.path.isdir(args.songmae_results) else os.path.dirname(songmae_csv)
        output_dir = os.path.join(base_dir, "plots_compare_models")
    os.makedirs(output_dir, exist_ok=True)

    mode_tag = _make_slug(args.mode or "all_modes")
    probe_tag = _make_slug(args.probe_mode or "all")
    out_png = os.path.join(output_dir, f"eval_f1_fer_{mode_tag}_{probe_tag}_songmae_vs_aves_joined.png")
    saved_png, saved_pdf = _save_plot_with_pdf(fig, out_png)
    plt.close(fig)

    print(f"Saved: {saved_png}")
    print(f"Saved: {saved_pdf}")


if __name__ == "__main__":
    main()
