#!/usr/bin/env python3
"""Generate UMAP plots for checkpoints every 50k steps."""

import re
import subprocess
import sys
from pathlib import Path


def _slug(value) -> str:
    text = str(value)
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in text)


PLOT_SCRIPT = Path(__file__).resolve().parents[1] / "src" / "plot_embedding.py"
RUN_DIR = Path("/home/george-vengrovski/Documents/projects/TinyBird/runs/Talapas_Run_2")
SPEC_DIR = Path("/media/george-vengrovski/disk2/avn_zf_data/zf_specs_64hop")
JSON_PATH = Path("/home/george-vengrovski/Documents/projects/TinyBird/files/zf_annotations.json")
NUM_TIMEBINS = 100_000
BIRD = "B145"
STEP_INCREMENT = 50_000
IMG_DIR = Path(__file__).resolve().parents[1] / "imgs"


def main() -> None:
    weights_dir = RUN_DIR / "weights"
    if not weights_dir.is_dir():
        raise SystemExit(f"Missing weights directory: {weights_dir}")

    pattern = re.compile(r"model_step_(\d+)\.pth$")
    checkpoints = []
    for ckpt in sorted(weights_dir.glob("model_step_*.pth")):
        match = pattern.match(ckpt.name)
        if not match:
            continue
        step = int(match.group(1))
        if step >= STEP_INCREMENT and step % STEP_INCREMENT == 0:
            checkpoints.append((step, ckpt))

    if not checkpoints:
        raise SystemExit("No checkpoints found at the requested interval.")

    base_cmd = [
        sys.executable,
        str(PLOT_SCRIPT),
        "--run_dir",
        str(RUN_DIR),
        "--spec_dir",
        str(SPEC_DIR),
        "--json_path",
        str(JSON_PATH),
        "--num_timebins",
        str(NUM_TIMEBINS),
        "--bird",
        BIRD,
    ]

    spec_slug = _slug(SPEC_DIR.name or "spec")
    run_slug = _slug(RUN_DIR.name or "run")
    base_img = IMG_DIR / f"{spec_slug}__{run_slug}__tb{NUM_TIMEBINS}.png"

    for step, ckpt_path in checkpoints:
        print(f"Generating UMAP for step {step}: {ckpt_path}")
        cmd = base_cmd + ["--checkpoint", str(ckpt_path)]
        subprocess.run(cmd, check=True)
        if base_img.exists():
            dest = IMG_DIR / f"{spec_slug}__{run_slug}__tb{NUM_TIMEBINS}__step{step:06d}.png"
            base_img.replace(dest)
            print(f"Saved {dest}")
        else:
            print(f"Warning: expected image not found at {base_img}")


if __name__ == "__main__":
    main()
