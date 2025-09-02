#!/usr/bin/env python3
# birdset_one_shot.py
#
# End-to-end helper to:
#   1) Pin Hugging Face caches to a chosen SSD
#   2) Download BirdSet shards (resume-safe)
#   3) Pre-extract all *.tar.gz via datasets extractor
#   4) Compute event-coverage stats
#   5) Cut only detected-event audio to OGG/FLAC into a target directory
#
# Example:
#   python birdset_one_shot.py \
#     --raw-root /media/george-vengrovski/disk1/birdset_raw \
#     --out-dir  /media/george-vengrovski/disk1/XCL_OGG \
#     --configs XCL --flat
#
# Requirements: datasets, huggingface_hub, ffmpeg in PATH

import os, sys, argparse, shutil, time, json, signal, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = "DBD-research-group/BirdSet"
REV  = "b0c14a03571a7d73d56b12c4b1db81952c4f7e64"  # known-stable
DEFAULT_CONFIGS = ["XCL"]
SR = 32000  # BirdSet train uses 32 kHz

# ------------------------ args ------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Download, pre-extract, summarize, and cut detected-event audio from BirdSet.")
    ap.add_argument("--raw-root", required=True, help="SSD root that will hold all HF caches and temp files.")
    ap.add_argument("--out-dir",  required=True, help="Directory for final detected-event snippets (OGG/FLAC).")
    ap.add_argument("--revision", default=REV, help="Commit SHA to pin.")
    ap.add_argument("--configs",  default=",".join(DEFAULT_CONFIGS),
                    help="Comma-separated configs to fetch (e.g., XCL,XCW).")
    ap.add_argument("--flat", action="store_true",
                    help="Write all snippets into a single flat directory instead of species subfolders.")
    ap.add_argument("--purge-old-home-cache", action="store_true",
                    help="Remove ~/.cache/huggingface/{hub,datasets,transformers} after verifying new paths.")
    ap.add_argument("--cleanup-partials", action="store_true",
                    help="Delete *.incomplete in downloads cache before resuming.")
    ap.add_argument("--skip-download", action="store_true", help="Skip dataset download step.")
    ap.add_argument("--skip-preextract", action="store_true", help="Skip pre-extract step.")
    ap.add_argument("--skip-coverage", action="store_true", help="Skip event-coverage computation.")
    ap.add_argument("--skip-snippets", action="store_true", help="Skip detected-event cutting step.")
    ap.add_argument("--qscale", type=int, default=5, help="libvorbis qscale if available (0..10).")
    ap.add_argument("--retries", type=int, default=2, help="ffmpeg retry attempts per cut.")
    ap.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 8)),
                    help="Parallel ffmpeg workers for cutting.")
    ap.add_argument("--no-validate", action="store_true",
                    help="Skip ffprobe validation of outputs (faster).")
    ap.add_argument("--fast-flac", action="store_true",
                    help="Use FLAC -compression_level 0 when falling back (fast).")
    ap.add_argument("--prefer-copy-first", action="store_true",
                    help="Try stream copy before re-encode (fast if sources permit).")
    return ap.parse_args()

# ------------------------ env + utils ------------------------

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def set_hf_env(raw_root: Path):
    hf_home  = raw_root / "hf"
    ds_cache = hf_home / "datasets"
    hub_cache = hf_home / "hub"  # not strictly needed for datasets, kept for clarity
    tf_cache = hf_home / "transformers"
    tmp_dir  = raw_root / "tmp"
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
    # Search both typical locations
    candidates = [ds_cache / "downloads"]
    for base in candidates:
        if base.exists():
            n = 0
            for p in base.rglob("*.incomplete"):
                try:
                    p.unlink()
                    n += 1
                except Exception:
                    pass
            print(f"Removed {n} partial files under {base}")

def ensure_ffmpeg():
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found in PATH")
    if not shutil.which("ffprobe"):
        sys.exit("ffprobe not found in PATH")

def have_encoder(name: str) -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return f" {name} " in out

def run_with_retries(cmd: list[str], retries: int) -> bool:
    for k in range(1, retries + 1):
        r = subprocess.run(cmd)
        if r.returncode == 0:
            return True
        time.sleep(min(30, 2 ** k))
    return False

def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in name)

def validate_audio(path: Path) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,sample_rate",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        return False
    lines = [x.strip() for x in r.stdout.splitlines() if x.strip()]
    return any(x == str(SR) for x in lines)

# ------------------------ pipeline steps ------------------------

def step_download(configs, revision, ds_cache: Path):
    from datasets import load_dataset, DownloadConfig
    # Put downloads under datasets cache to match extraction step
    dcfg = DownloadConfig(cache_dir=str(ds_cache), resume_download=True)

    for name in configs:
        print(f"\n== download {name} =="); sys.stdout.flush()
        try:
            _ = load_dataset(
                REPO, name,
                trust_remote_code=True,
                revision=revision,
                cache_dir=str(ds_cache),      # Arrow/extracted cache
                download_config=dcfg          # .tar.gz go under ds_cache/downloads
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
    try:
        from datasets.utils.extract import extract_archive
    except Exception:
        from datasets.utils.file_utils import extract_archive  # type: ignore

    downloads = ds_cache / "downloads"
    extracted = ds_cache / "extracted"
    ensure_dirs(extracted)

    tars = sorted(downloads.rglob("*.tar.gz")) if downloads.exists() else []
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

def step_coverage(configs, revision, ds_cache: Path):
    from datasets import load_dataset, DownloadConfig
    dcfg = DownloadConfig(cache_dir=str(ds_cache), resume_download=True)

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

def ffmpeg_cut(src: str, start: float, end: float, dst_base: Path,
               qscale: int, retries: int, validate: bool,
               fast_flac: bool, prefer_copy_first: bool) -> Path | None:
    dur = max(0.02, float(end) - float(start))
    dst_base.parent.mkdir(parents=True, exist_ok=True)

    # Optionally try stream copy first (very fast if compatible)
    if prefer_copy_first:
        dst = dst_base.with_suffix(".ogg")
        cmd = ["ffmpeg","-hide_banner","-loglevel","error",
               "-ss", f"{start:.3f}","-t", f"{dur:.3f}",
               "-i", src, "-vn", "-c","copy", "-y", str(dst)]
        if run_with_retries(cmd, retries) and dst.exists() and dst.stat().st_size > 0 and (True if not validate else validate_audio(dst)):
            return dst

    # Vorbis at 32 kHz if available
    if have_encoder("libvorbis"):
        dst = dst_base.with_suffix(".ogg")
        cmd = ["ffmpeg","-hide_banner","-loglevel","error",
               "-ss", f"{start:.3f}","-t", f"{dur:.3f}",
               "-i", src, "-vn", "-c:a","libvorbis","-qscale:a", str(qscale),
               "-ar", str(SR), "-y", str(dst)]
        if run_with_retries(cmd, retries) and dst.exists() and dst.stat().st_size > 0 and (True if not validate else validate_audio(dst)):
            return dst

    # Fallback: FLAC at 32 kHz (lossless)
    dst = dst_base.with_suffix(".flac")
    cmd = ["ffmpeg","-hide_banner","-loglevel","error",
           "-ss", f"{start:.3f}","-t", f"{dur:.3f}",
           "-i", src, "-vn", "-c:a","flac","-compression_level","0" if fast_flac else "5",
           "-ar", str(SR), "-y", str(dst)]
    if run_with_retries(cmd, retries) and dst.exists() and dst.stat().st_size > 0 and (True if not validate else validate_audio(dst)):
        return dst

    return None

def _cut_one(task):
    # task tuple: (src, st, en, out_base, qscale, retries, validate, fast_flac, prefer_copy_first, cfg, sp)
    (src, st, en, out_base, qscale, retries, validate, fast_flac, prefer_copy_first, cfg, sp) = task
    dst = ffmpeg_cut(src, st, en, Path(out_base), qscale, retries, validate, fast_flac, prefer_copy_first)
    if dst:
        return {"config": cfg, "species": sp, "src": src, "start": float(st), "end": float(en), "dst": str(dst)}
    return None

def step_snippets(configs, revision, ds_cache: Path, out_dir: Path, flat: bool, qscale: int, retries: int,
                  workers: int, validate: bool, fast_flac: bool, prefer_copy_first: bool):
    ensure_ffmpeg()
    from datasets import load_dataset, DownloadConfig
    dcfg = DownloadConfig(cache_dir=str(ds_cache), resume_download=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "snippets_manifest.jsonl"
    fails_path = out_dir / "snippets_failures.log"

    total_src = kept = written = 0
    max_outstanding = workers * 4

    with open(manifest_path, "a", encoding="utf-8") as man, open(fails_path, "a", encoding="utf-8") as flog:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for cfg in configs:
                print(f"\n== snippets {cfg} ==")
                try:
                    ds = load_dataset(
                        REPO, cfg, split="train",
                        trust_remote_code=True,
                        revision=revision,
                        cache_dir=str(ds_cache),
                        download_config=dcfg,
                        download_mode="reuse_dataset_if_exists",
                    )
                except Exception as e:
                    print(f"{cfg}: load failed -> {e}")
                    continue

                to_label = ds.features["ebird_code"].int2str if "ebird_code" in ds.features else (lambda i: "unknown")

                futures = []
                for ex in ds:
                    total_src += 1
                    events = ex.get("detected_events") or []
                    if not events:
                        continue
                    kept += 1

                    src_path = ex["audio"]["path"] if "audio" in ex and isinstance(ex["audio"], dict) else None
                    if not src_path or not Path(src_path).exists():
                        flog.write(f"missing_audio\t{src_path}\n"); continue

                    sp = sanitize(to_label(ex.get("ebird_code", 0)))
                    base = sanitize(Path(src_path).stem)

                    for j, span in enumerate(events):
                        try:
                            st, en = float(span[0]), float(span[1])
                        except Exception:
                            flog.write(f"bad_event\t{src_path}\t{span}\n"); continue

                        if flat:
                            # Single folder. Keep names globally unique.
                            out_base = out_dir / f"{cfg}_{sp}_{base}_ev{j:03d}"
                        else:
                            out_base = out_dir / cfg / sp / f"{base}_ev{j:03d}"

                        if out_base.with_suffix(".ogg").exists() or out_base.with_suffix(".flac").exists():
                            continue
                        task = (src_path, st, en, str(out_base), qscale, retries, validate, fast_flac, prefer_copy_first, cfg, sp)
                        futures.append(executor.submit(_cut_one, task))
                        if len(futures) >= max_outstanding:
                            for f in as_completed(futures[:workers]):
                                rec = f.result()
                                if rec:
                                    man.write(json.dumps(rec) + "\n"); written += 1
                                else:
                                    pass
                            del futures[:workers]
                # drain remaining
                for f in as_completed(futures):
                    rec = f.result()
                    if rec:
                        man.write(json.dumps(rec) + "\n"); written += 1

    print(f"\nDone. source items: {total_src}  with_events: {kept}  snippets_written: {written}")
    print(f"Manifest: {manifest_path}")
    print(f"Failures: {fails_path}")
    print(f"Snippets root: {out_dir}")

# ------------------------ main ------------------------

def main():
    # clean Ctrl-C handling
    signal.signal(signal.SIGINT, lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))

    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    out_dir  = Path(args.out_dir).expanduser().resolve()
    ds_cache, hub_cache, tmp_dir = set_hf_env(raw_root)

    print("Raw root:", raw_root)
    print("Out dir :", out_dir)
    print("Revision:", args.revision)
    verify_paths(ds_cache, hub_cache, tmp_dir)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if not configs:
        print("no configs given", file=sys.stderr)
        sys.exit(2)

    if args.cleanup_partials:
        cleanup_partials(ds_cache)

    if not args.skip_download:
        step_download(configs, args.revision, ds_cache)
    else:
        print("skip-download: enabled")

    if not args.skip_preextract:
        step_preextract(ds_cache)
    else:
        print("skip-preextract: enabled")

    if not args.skip_coverage:
        step_coverage(configs, args.revision, ds_cache)
    else:
        print("skip-coverage: enabled")

    if not args.skip_snippets:
        step_snippets(configs, args.revision, ds_cache, out_dir, args.flat,
                      args.qscale, args.retries, args.workers,
                      validate=(not args.no_validate), fast_flac=args.fast_flac,
                      prefer_copy_first=args.prefer_copy_first)
    else:
        print("skip-snippets: enabled")

    maybe_purge_old_home_cache(args.purge_old_home_cache)
    print("\nAll caches and temp live under:", raw_root)
    print("Final snippets under:", out_dir)

if __name__ == "__main__":
    main()
