import argparse
import json
import re
import sys
from pathlib import Path


_CHUNK_MS_RE = re.compile(r"^(?P<base>.+)__ms_(?P<start>\d+)_(?P<end>\d+)$")


def _base_stem(stem: str) -> str:
    base = stem
    while True:
        match = _CHUNK_MS_RE.match(base)
        if not match:
            break
        base = match.group("base")
    return base


def build_manifest(train_dir, val_dir, wav_root, out_path, wav_exts):
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    wav_root = Path(wav_root)
    out_path = Path(out_path)

    exts = tuple(e.strip().lower() for e in wav_exts.split(",") if e.strip())

    stems = set()
    for root in (train_dir, val_dir):
        for path in root.glob("*.npy"):
            stems.add(_base_stem(path.stem))

    if not stems:
        raise SystemExit(f"No .npy files found in {train_dir} or {val_dir}")

    index = {}
    dupes = {}
    for path in wav_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        stem = path.stem
        if stem in index:
            dupes.setdefault(stem, [index[stem]]).append(path)
            continue
        index[stem] = path

    missing = [stem for stem in sorted(stems) if stem not in index]
    ambiguous = [stem for stem in sorted(stems) if stem in dupes]

    if missing or ambiguous:
        if missing:
            print(
                f"Missing wavs for {len(missing)} stems (sample: {missing[:5]})",
                file=sys.stderr,
            )
        if ambiguous:
            print(
                f"Duplicate wav stems for {len(ambiguous)} stems (sample: {ambiguous[:5]})",
                file=sys.stderr,
            )
        raise SystemExit(1)

    manifest = {stem: str(index[stem]) for stem in sorted(stems)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote wav manifest: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a wav manifest from train/val spec dirs."
    )
    parser.add_argument("--train_dir", required=True, help="Train .npy directory.")
    parser.add_argument("--val_dir", required=True, help="Val .npy directory.")
    parser.add_argument("--wav_root", required=True, help="Root directory for wav files.")
    parser.add_argument("--out_path", required=True, help="Output manifest path.")
    parser.add_argument(
        "--wav_exts",
        default=".wav,.flac,.ogg,.mp3",
        help="Comma-separated wav extensions to index.",
    )
    args = parser.parse_args()

    build_manifest(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        wav_root=args.wav_root,
        out_path=args.out_path,
        wav_exts=args.wav_exts,
    )


if __name__ == "__main__":
    main()
