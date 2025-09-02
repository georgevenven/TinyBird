# birdset_event_coverage_local.py
import os
from pathlib import Path
from datasets import load_dataset, DownloadConfig

REPO = "DBD-research-group/BirdSet"
DEFAULT_REV = "b0c14a03571a7d73d56b12c4b1db81952c4f7e64"
CONFIGS = ["XCL"]

def main():
    # Hardcoded cache directory path
    cache = Path("/media/george-vengrovski/disk1/XCL_RAW")
    revision = DEFAULT_REV
    configs = CONFIGS

    os.environ["HF_DATASETS_CACHE"] = str(cache)               # where .tar.gz and extracted files live
    dlcfg = DownloadConfig(cache_dir=str(cache))

    total_n = total_k = 0
    for name in configs:
        try:
            ds = load_dataset(
                REPO, name, split="train",
                trust_remote_code=True, revision=revision,
                download_config=dlcfg,
                download_mode="reuse_dataset_if_exists",  # never redownload
            )
        except Exception as e:
            print(f"{name}: load failed -> {e}")
            continue

        n = len(ds)
        k = sum(1 for ex in ds if ex.get("detected_events") and len(ex["detected_events"]) > 0)
        pct = 100.0 * k / n if n else 0.0
        print(f"{name:4s}  {k:7d}/{n:7d}  {pct:5.1f}%")
        total_n += n; total_k += k

    if total_n:
        print(f"\nOverall  {total_k}/{total_n}  {100.0*total_k/total_n:5.1f}%")

if __name__ == "__main__":
    main()
