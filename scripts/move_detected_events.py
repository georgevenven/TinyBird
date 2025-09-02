#!/usr/bin/env python3
# make_birdset_snippets.py
# deps: pip install "datasets<4" huggingface_hub
# also requires: ffmpeg in PATH

import os, subprocess, time, json, shutil, sys
from pathlib import Path
from datasets import load_dataset, DownloadConfig

# ====== EDIT THESE ======
CACHE_DIR = Path("/media/george-vengrovski/disk1/birdset_raw_download/datasets")  # your HF datasets cache (has downloads/)
DEST_DIR  = Path("/media/george-vengrovski/disk1/birdset_snippets")               # output root
CONFIGS   = ["XCM"]                                                                # add others if needed
REVISION  = "b0c14a03571a7d73d56b12c4b1db81952c4f7e64"                             # pinned BirdSet commit
RETRIES   = 4
# ========================

SR = 32000  # target sample rate (BirdSet train is 32 kHz)

def ensure_ffmpeg():
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found in PATH")
    if not shutil.which("ffprobe"):
        sys.exit("ffprobe not found in PATH")

def have_encoder(name: str) -> bool:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    ).stdout
    return f" {name} " in out

def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-+" else "_" for c in name)

def validate_audio(path: Path) -> bool:
    # Decode a few packets; if ffprobe can read stream and reports sample_rate, accept
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,sample_rate",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        return False
    lines = [x.strip() for x in r.stdout.splitlines() if x.strip()]
    # Expect codec then samplerate; tolerate order
    sr_ok = any(x == str(SR) for x in lines)
    return len(lines) >= 1 and sr_ok

def ffmpeg_cut(src: str, start: float, end: float, dst_base: Path) -> Path | None:
    """Cut [start,end) from src into dst_base with appropriate extension and codec."""
    dur = max(0.02, float(end) - float(start))
    dst_base.parent.mkdir(parents=True, exist_ok=True)

    # Mode 1: Vorbis at 32 kHz if available (keeps SR and Ogg container)
    if have_encoder("libvorbis"):
        dst = dst_base.with_suffix(".ogg")
        cmd = ["ffmpeg","-hide_banner","-loglevel","error",
               "-ss", f"{start:.3f}","-t", f"{dur:.3f}",
               "-i", src, "-vn", "-c:a","libvorbis","-qscale:a","5",
               "-ar", str(SR), "-y", str(dst)]
        if run_with_retries(cmd) and dst.exists() and dst.stat().st_size > 0 and validate_audio(dst):
            return dst

    # Mode 2: stream copy (no re-encode); may fail on Ogg page boundaries
    dst = dst_base.with_suffix(".ogg")
    cmd = ["ffmpeg","-hide_banner","-loglevel","error",
           "-ss", f"{start:.3f}","-t", f"{dur:.3f}",
           "-i", src, "-vn", "-c","copy", "-y", str(dst)]
    if run_with_retries(cmd) and dst.exists() and dst.stat().st_size > 0 and validate_audio(dst):
        return dst

    # Mode 3: FLAC at 32 kHz (lossless, preserves SR, different codec)
    dst = dst_base.with_suffix(".flac")
    cmd = ["ffmpeg","-hide_banner","-loglevel","error",
           "-ss", f"{start:.3f}","-t", f"{dur:.3f}",
           "-i", src, "-vn", "-c:a","flac","-compression_level","5",
           "-ar", str(SR), "-y", str(dst)]
    if run_with_retries(cmd) and dst.exists() and dst.stat().st_size > 0 and validate_audio(dst):
        return dst

    return None

def run_with_retries(cmd: list[str]) -> bool:
    for k in range(1, RETRIES + 1):
        r = subprocess.run(cmd)
        if r.returncode == 0:
            return True
        time.sleep(min(30, 2 ** k))
    return False

def main():
    ensure_ffmpeg()
    # Force local cache use only
    os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR)
    dlcfg = DownloadConfig(cache_dir=str(CACHE_DIR))

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DEST_DIR / "snippets_manifest.jsonl"
    fails_path = DEST_DIR / "snippets_failures.log"

    total_src = kept = written = 0

    with open(manifest_path, "a", encoding="utf-8") as man, open(fails_path, "a", encoding="utf-8") as flog:
        for cfg in CONFIGS:
            print(f"\n== {cfg} ==")
            ds = load_dataset(
                "DBD-research-group/BirdSet", cfg,
                split="train", trust_remote_code=True,
                revision=REVISION, download_config=dlcfg,
                download_mode="reuse_dataset_if_exists",  # do not redownload
            )
            to_label = ds.features["ebird_code"].int2str
            for ex in ds:
                total_src += 1
                events = ex.get("detected_events") or []
                if not events:
                    continue
                kept += 1

                src_path = ex["audio"]["path"]
                if not src_path or not Path(src_path).exists():
                    flog.write(f"missing_audio\t{src_path}\n"); continue

                sp = sanitize(to_label(ex["ebird_code"]))
                base = sanitize(Path(src_path).stem)

                for j, (st, en) in enumerate(events):
                    out_base = DEST_DIR / cfg / sp / f"{base}_ev{j:03d}"
                    # Skip if any prior success exists (.ogg or .flac)
                    if out_base.with_suffix(".ogg").exists() or out_base.with_suffix(".flac").exists():
                        continue
                    dst = ffmpeg_cut(src_path, float(st), float(en), out_base)
                    if dst:
                        rec = {
                            "config": cfg, "species": sp, "src": src_path,
                            "start": float(st), "end": float(en),
                            "dst": str(dst)
                        }
                        man.write(json.dumps(rec) + "\n")
                        written += 1
                    else:
                        flog.write(f"cut_failed\t{src_path}\t{st}\t{en}\n")

    print(f"\nDone. source items: {total_src}  with_events: {kept}  snippets_written: {written}")
    print(f"Manifest: {manifest_path}")
    print(f"Failures: {fails_path}")
    print(f"Snippets root: {DEST_DIR}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted."); sys.exit(130)
