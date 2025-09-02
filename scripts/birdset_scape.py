#!/usr/bin/env python3
# birdset_fetch_all.py
# Usage:
#   python birdset_fetch_all.py --root /mnt/ssdpath [--revision SHA] [--configs XCL,XCW] [--purge-old-home-cache]
#
# This script pins all Hugging Face caches and temp dirs to a target SSD,
# then downloads BirdSet raw recordings.

import os, sys, argparse, shutil
from pathlib import Path

REPO = "DBD-research-group/BirdSet"
REV  = "b0c14a03571a7d73d56b12c4b1db81952c4f7e64"
DEFAULT_CONFIGS = ["XCL"]  # add more if needed

def parse_args():
    ap = argparse.ArgumentParser(description="Download ALL BirdSet raw recordings to a non-default SSD.")
    ap.add_argument("--root", required=True, help="Root dir on target SSD (e.g., /mnt/bigssd)")
    ap.add_argument("--revision", default=REV, help="Commit SHA to pin (default = known-stable)")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                    help="Comma-separated dataset configs to fetch (default: XCL)")
    ap.add_argument("--purge-old-home-cache", action="store_true",
                    help="Remove ~/.cache/huggingface/{hub,datasets,transformers} after verifying new paths.")
    return ap.parse_args()

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def set_hf_env(root: Path):
    hf_home = root / "hf"
    ds_cache = hf_home / "datasets"
    hub_cache = hf_home / "hub"
    tf_cache = hf_home / "transformers"
    tmp_dir  = root / "tmp"

    ensure_dirs(hf_home, ds_cache, hub_cache, tf_cache, tmp_dir)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(ds_cache)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)  # alias some libs read
    os.environ["TRANSFORMERS_CACHE"] = str(tf_cache)
    os.environ["TMPDIR"] = str(tmp_dir)

    return ds_cache, hub_cache, tmp_dir

def verify_paths(ds_cache, hub_cache, tmp_dir):
    import datasets, huggingface_hub
    print("HF_HUB_CACHE    =", huggingface_hub.constants.HF_HUB_CACHE)
    print("HF_DATASETS_DIR =", datasets.config.HF_DATASETS_CACHE)
    print("TMPDIR          =", tmp_dir)
    if not str(hub_cache).startswith(str(huggingface_hub.constants.HF_HUB_CACHE)):
        print("warning: hub cache mismatch", file=sys.stderr)
    if not str(ds_cache).startswith(str(datasets.config.HF_DATASETS_CACHE)):
        print("warning: datasets cache mismatch", file=sys.stderr)

def maybe_purge_old_home_cache(enabled: bool):
    if not enabled:
        return
    home = Path.home() / ".cache" / "huggingface"
    if not home.exists():
        return
    targets = [home / "hub", home / "datasets", home / "transformers"]
    for t in targets:
        if t.exists():
            print("Removing:", t)
            shutil.rmtree(t, ignore_errors=True)

def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    ds_cache, hub_cache, tmp_dir = set_hf_env(root)

    # Import AFTER env is set so caches are read from env.
    from datasets import load_dataset, DownloadConfig

    print("Root:", root)
    print("Revision:", args.revision)
    verify_paths(ds_cache, hub_cache, tmp_dir)

    # Hub download cache (archives/scripts) → hub_cache
    dcfg = DownloadConfig(cache_dir=str(hub_cache))

    # Normalize configs
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if not configs:
        print("no configs given", file=sys.stderr)
        sys.exit(2)

    for name in configs:
        print(f"\n== {name} =="); sys.stdout.flush()
        try:
            _ = load_dataset(
                REPO, name,
                trust_remote_code=True,
                revision=args.revision,
                cache_dir=str(ds_cache),      # Arrow cache
                download_config=dcfg          # Hub cache
            )
            print("downloaded: all defined splits for", name)
        except Exception as e:
            print("multi-split load failed:", e)
            print("fallback → downloading TRAIN only for", name); sys.stdout.flush()
            try:
                _ = load_dataset(
                    REPO, name, split="train",
                    trust_remote_code=True,
                    revision=args.revision,
                    cache_dir=str(ds_cache),
                    download_config=dcfg
                )
                print("downloaded: train for", name)
            except Exception as e2:
                print("failed:", name, "->", e2)

    maybe_purge_old_home_cache(args.purge_old_home_cache)
    print("\nDone. All files and caches are under:", root)

if __name__ == "__main__":
    main()
