import argparse
import json
from pathlib import Path


def get_ms_per_timebin(spec_dir: Path) -> str:
    audio_path = spec_dir / "audio_params.json"
    if not audio_path.exists():
        return ""
    try:
        data = json.loads(audio_path.read_text(encoding="utf-8"))
        sr = float(data.get("sr", 0.0))
        hop = float(data.get("hop_size", 0.0))
        if sr > 0 and hop > 0:
            return str(hop / sr * 1000.0)
    except Exception:
        return ""
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Compute ms_per_timebin from spec_dir/audio_params.json."
    )
    parser.add_argument("--spec_dir", required=True, help="Spec directory.")
    args = parser.parse_args()

    spec_dir = Path(args.spec_dir)
    value = get_ms_per_timebin(spec_dir)
    if value:
        print(value)


if __name__ == "__main__":
    main()
