#!/usr/bin/env python3
# birdset_fetch_extract_coverage.py
#
# One-shot helper to:
#   1) Pin Hugging Face caches to a fast SSD
#   2) Download BirdSet shards (resume-safe)
#   3) Pre-extract all *.tar.gz with the datasets extractor
#   4) Compute event-coverage stats
#
# Usage examples:
#   python birdset_fetch_extract_coverage.py --root /mnt/ssdpath
#   python birdset_fetch_extract_coverage.py --root /mnt/ssdpath --configs XCL,XCW
#   python birdset_fetch_extract_coverage.py --root /mnt/ssdpath --skip-coverage
#   python birdset_fetch_extract_coverage.py --root /mnt/ssdpath --cleanup-partials
#
# Requires: datasets, huggingface_hub

import os
import sys
import argparse
import shutil
import signal
from pathlib import Path

REPO = "DBD-research-group/BirdSet"
REV  = "b0c14a03571a7d73d56b12c4b1db81952c4f7e64"  # known-stable
DEFAULT_CONFIGS = ["XCL"]

# ------------------------ args ------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Fetch, pre-extract, and summarize BirdSet using SSD caches.")
    ap.add_argument("--root", required=True, help="SSD root (e.g., /mnt/ssdpath)")
    ap.add_argument("--revision", default=REV, help="Commit SHA to pin")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                    help="Comma-separated configs to fetch (e.g., XCL,XCW,POW)")
    ap.add_argument("--purge-old-home-cache", action="store_true",
                    help="Remove ~/.cache/huggingface/{hub,datasets,transformers} after verifying new paths.")
    ap.add_argument("--cleanup-partials", action="store_true",
                    help="Delete *.incomplete in SSD downloads cache before resuming.")
    ap.add_argument("--skip-download", action="store_true", help="Skip dataset download step.")
    ap.add_argument("--skip-preextract", action="store_true", help="Skip pre-extract step.")
    ap.add_argument("--skip-coverage", action="store_true", help="Skip event-coverage computation.")
    return ap.parse_args()

# ------------------------ env + utils ------------------------

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
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(tf_cache)
    os.environ["TMPDIR"] = str(tmp_dir)

    return ds_cache, hub_cache, tmp_dir

def verify_paths(ds_cache: Path, hub_cache: Path, tmp_dir: Path):
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
    for sub in ["hub", "datasets", "transformers"]:
        t = home / sub
        if t.exists():
            print("Removing:", t)
            shutil.rmtree(t, ignore_errors=True)

def cleanup_partials(ds_cache: Path):
    downloads = ds_cache / "downloads"
    if downloads.exists():
        n = 0
        for p in downloads.rglob("*.incomplete"):
            try:
                p.unlink()
                n += 1
            except Exception:
                pass
        print(f"Removed {n} partial files")

# ------------------------ steps ------------------------

def step_download(configs, revision, ds_cache: Path, hub_cache: Path):
    from datasets import load_dataset, DownloadConfig
    dcfg = DownloadConfig(cache_dir=str(hub_cache), resume_download=True)

    for name in configs:
        print(f"\n== download {name} =="); sys.stdout.flush()
        try:
            _ = load_dataset(
                REPO, name,
                trust_remote_code=True,
                revision=revision,
                cache_dir=str(ds_cache),      # Arrow/extracted cache
                download_config=dcfg          # Hub downloads cache
            )
            print("downloaded: all defined splits for", name)
        except Exception as e:
            print("multi-split load failed:", e)
            print("fallback → downloading TRAIN only for", name); sys.stdout.flush()
            try:
                _ = load_dataset(
                    REPO, name, split="train",
                    trust_remote_code=True,
                    revision=revision,
                    cache_dir=str(ds_cache),
                    download_config=dcfg
                )
                print("downloaded: train for", name)
            except Exception as e2:
                print("failed:", name, "->", e2)

def step_preextract(ds_cache: Path):
    # Use datasets' extractor so paths match the library's hashing scheme
    try:
        from datasets.utils.extract import extract_archive
    except Exception:
        # Fallback location for very old versions
        from datasets.utils.file_utils import extract_archive  # type: ignore

    downloads = ds_cache / "downloads"
    extracted = ds_cache / "extracted"
    ensure_dirs(extracted)

    tars = sorted(downloads.rglob("*.tar.gz"))
    if not tars:
        print("no tarballs found under", downloads)
        return

    for p in tars:
        try:
            out = extract_archive(str(p), str(extracted))
            print(f"ok: {p.name} -> {out}")
        except KeyboardInterrupt:
            print("\ninterrupt during extraction"); raise
        except Exception as e:
            print(f"extract failed: {p} -> {e}")

def step_coverage(configs, revision, ds_cache: Path, hub_cache: Path):
    from datasets import load_dataset, DownloadConfig
    dcfg = DownloadConfig(cache_dir=str(hub_cache), resume_download=True)

    total_n = total_k = 0
    for name in configs:
        print(f"\n== coverage {name} ==")
        try:
            ds = load_dataset(
                REPO, name, split="train",
                trust_remote_code=True,
                revision=revision,
                cache_dir=str(ds_cache),
                download_config=dcfg,
                download_mode="reuse_dataset_if_exists",
                verification_mode="no_checks"
            )
        except Exception as e:
            print(f"{name}: load failed -> {e}")
            continue

        n = len(ds)
        k = 0
        for ex in ds:
            ev = ex.get("detected_events")
            if ev and len(ev) > 0:
                k += 1
        pct = 100.0 * k / n if n else 0.0
        print(f"{name:4s}  {k:7d}/{n:7d}  {pct:5.1f}%")
        total_n += n; total_k += k

    if total_n:
        print(f"\nOverall  {total_k}/{total_n}  {100.0*total_k/total_n:5.1f}%")

# ------------------------ main ------------------------

def main():
    # clean Ctrl-C handling
    signal.signal(signal.SIGINT, lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))

    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    ds_cache, hub_cache, tmp_dir = set_hf_env(root)

    print("Root:", root)
    print("Revision:", args.revision)
    verify_paths(ds_cache, hub_cache, tmp_dir)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if not configs:
        print("no configs given", file=sys.stderr)
        sys.exit(2)

    if args.cleanup_partials:
        cleanup_partials(ds_cache)

    if not args.skip_download:
        step_download(configs, args.revision, ds_cache, hub_cache)
    else:
        print("skip-download: enabled")

    if not args.skip_preextract:
        step_preextract(ds_cache)
    else:
        print("skip-preextract: enabled")

    if not args.skip_coverage:
        step_coverage(configs, args.revision, ds_cache, hub_cache)
    else:
        print("skip-coverage: enabled")

    maybe_purge_old_home_cache(args.purge_old_home_cache)
    print("\nDone. All files and caches are under:", root)

if __name__ == "__main__":
    main()
