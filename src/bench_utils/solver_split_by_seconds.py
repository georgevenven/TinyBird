import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


CHUNK_MS_RE = re.compile(r"^(?P<base>.+)__ms_(?P<start>\d+)_(?P<end>\d+)$")


def parse_chunk(stem: str):
    base = stem
    start = None
    end = None
    while True:
        m = CHUNK_MS_RE.match(base)
        if not m:
            break
        base = m.group("base")
        start = int(m.group("start"))
        end = int(m.group("end"))
    return base, start, end


def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def solve_and_split(
    pool_dir: Path,
    annotation_file: Path,
    test_dir: Path,
    train_dir: Path,
    train_seconds: float | str,
    test_ratio: float = 0.2,
    seed: int = 42,
    feasibility_json: Path | None = None,
):
    def update_info(extra):
        if feasibility_json is None:
            return
        base = {}
        if feasibility_json.exists():
            try:
                base = load_json(feasibility_json)
            except Exception:
                base = {}
        base.update(extra)
        save_json(feasibility_json, base)

    def fail(reason: str, status: int = 2, message: str | None = None):
        update_info(
            {
                "solver_feasible": False,
                "solver_status": int(status),
                "solver_message": str(message or reason),
                "failure_reason": str(reason),
            }
        )
        raise RuntimeError(message or reason)

    audio_params_path = pool_dir / "audio_params.json"
    if not audio_params_path.exists():
        fail(f"Missing {audio_params_path}", status=2)
    audio_params = load_json(audio_params_path)
    sr = float(audio_params["sr"])
    hop = float(audio_params["hop_size"])

    def ms_to_timebins(ms):
        return int(round((float(ms) / 1000.0) * sr / hop))

    def timebins_to_ms(tb):
        return int(round(float(tb) * hop / sr * 1000.0))

    def move_to_dir(path: Path, target_dir: Path):
        target_dir.mkdir(parents=True, exist_ok=True)
        dst = target_dir / path.name
        if dst.exists() and dst != path:
            fail(
                "name_collision",
                status=3,
                message=f"Destination exists during move: {dst}",
            )
        shutil.move(str(path), str(dst))
        return dst

    spec_files = sorted(pool_dir.glob("*.npy"))
    if seed is not None:
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(len(spec_files))
        spec_files = [spec_files[int(i)] for i in perm]
    if not spec_files:
        fail("No pool .npy files to split", status=2)

    durations_bins = []
    for p in spec_files:
        arr = np.load(p, mmap_mode="r")
        durations_bins.append(int(arr.shape[1]))
    durations_bins = np.array(durations_bins, dtype=float)
    pool_bins = int(np.sum(durations_bins))
    test_target_bins = int(round(pool_bins * float(test_ratio)))
    train_seconds_token = str(train_seconds).strip()
    if train_seconds_token.upper() == "MAX":
        train_target_bins = int(pool_bins - test_target_bins)
    else:
        try:
            train_seconds_value = float(train_seconds_token)
        except ValueError:
            fail(
                "invalid_train_seconds",
                status=2,
                message=f"Unsupported --train_seconds value: {train_seconds!r}",
            )
        if train_seconds_value < 0:
            fail(
                "invalid_train_seconds",
                status=2,
                message=f"--train_seconds must be non-negative or MAX, got: {train_seconds_value}",
            )
        train_target_bins = int(round(train_seconds_value * sr / hop))

    ann = load_json(annotation_file)
    units_by_base = defaultdict(list)
    all_units = set()
    for rec in ann.get("recordings", []):
        base = Path(rec["recording"]["filename"]).stem
        for event in rec.get("detected_events", []):
            for unit in event.get("units", []):
                uid = int(unit["id"])
                on = float(unit["onset_ms"])
                off = float(unit["offset_ms"])
                units_by_base[base].append((uid, on, off))
                all_units.add(uid)

    def file_unit_intervals(path: Path, timebins: int):
        base, chunk_start, chunk_end = parse_chunk(path.stem)
        chunk_offset = float(chunk_start or 0.0)
        file_ms = float(timebins_to_ms(timebins))
        chunk_end_abs = float(chunk_end) if chunk_end is not None else chunk_offset + file_ms
        out = defaultdict(list)
        for uid, on_abs, off_abs in units_by_base.get(base, []):
            if off_abs <= chunk_offset or on_abs >= chunk_end_abs:
                continue
            on_local = max(on_abs, chunk_offset) - chunk_offset
            off_local = min(off_abs, chunk_end_abs) - chunk_offset
            on_local = max(0.0, min(on_local, file_ms))
            off_local = max(0.0, min(off_local, file_ms))
            if off_local > on_local:
                out[int(uid)].append((on_local, off_local))
        return out

    files_by_unit = defaultdict(list)
    for i, p in enumerate(spec_files):
        t = int(durations_bins[i])
        file_u = set(file_unit_intervals(p, t).keys())
        for uid in file_u:
            files_by_unit[uid].append(i)

    n = len(spec_files)
    num_vars = 2 * n + 4
    e_tp, e_tn, e_rp, e_rn = 2 * n, 2 * n + 1, 2 * n + 2, 2 * n + 3

    c = np.zeros(num_vars, dtype=float)
    c[e_tp] = 1.0
    c[e_tn] = 1.0
    c[e_rp] = 1.0
    c[e_rn] = 1.0

    A = []
    lb = []
    ub = []

    # Disjoint file assignment.
    for i in range(n):
        row = np.zeros(num_vars, dtype=float)
        row[i] = 1.0
        row[n + i] = 1.0
        A.append(row)
        lb.append(-np.inf)
        ub.append(1.0)

    # Require unit coverage in both splits.
    for uid in sorted(all_units):
        idxs = files_by_unit.get(uid, [])
        if not idxs:
            continue
        row_t = np.zeros(num_vars, dtype=float)
        row_r = np.zeros(num_vars, dtype=float)
        for i in idxs:
            row_t[i] = 1.0
            row_r[n + i] = 1.0
        A.append(row_t)
        lb.append(1.0)
        ub.append(np.inf)
        A.append(row_r)
        lb.append(1.0)
        ub.append(np.inf)

    # Minimize absolute duration errors (test + train).
    row_test = np.zeros(num_vars, dtype=float)
    row_test[:n] = durations_bins
    row_test[e_tp] = -1.0
    row_test[e_tn] = 1.0
    A.append(row_test)
    lb.append(float(test_target_bins))
    ub.append(float(test_target_bins))

    row_train = np.zeros(num_vars, dtype=float)
    row_train[n:2 * n] = durations_bins
    row_train[e_rp] = -1.0
    row_train[e_rn] = 1.0
    A.append(row_train)
    lb.append(float(train_target_bins))
    ub.append(float(train_target_bins))

    constraints = LinearConstraint(np.vstack(A), np.array(lb, dtype=float), np.array(ub, dtype=float))
    integrality = np.zeros(num_vars, dtype=int)
    integrality[: 2 * n] = 1
    bounds = Bounds(
        np.zeros(num_vars, dtype=float),
        np.concatenate([np.ones(2 * n, dtype=float), np.full(4, np.inf, dtype=float)]),
    )

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    if res.status != 0 or res.x is None:
        fail(
            "solver_infeasible",
            status=2,
            message=f"status={res.status} message={res.message}",
        )

    x = res.x
    test_idx = [i for i in range(n) if x[i] > 0.5]
    train_idx = [i for i in range(n) if x[n + i] > 0.5]

    test_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    for i in test_idx:
        move_to_dir(spec_files[i], test_dir)
    for i in train_idx:
        move_to_dir(spec_files[i], train_dir)

    shutil.copy2(audio_params_path, test_dir / "audio_params.json")
    shutil.copy2(audio_params_path, train_dir / "audio_params.json")

    def crop_file_to_bins(path: Path, keep_bins: int, out_dir: Path, req_start_ms=None, req_end_ms=None):
        arr = np.load(path, mmap_mode="r")
        total_bins = int(arr.shape[1])
        if keep_bins <= 0 or keep_bins > total_bins:
            fail("invalid_keep_bins", status=3, message=f"path={path} keep={keep_bins} total={total_bins}")

        if req_start_ms is None or req_end_ms is None or keep_bins == total_bins:
            start_bin = 0
        else:
            req_start_bin = max(0, min(ms_to_timebins(req_start_ms), total_bins - 1))
            req_end_bin = max(req_start_bin + 1, min(ms_to_timebins(req_end_ms), total_bins))
            req_span = req_end_bin - req_start_bin
            if keep_bins < req_span:
                fail(
                    "required_span_exceeds_keep",
                    status=3,
                    message=f"path={path} req_span={req_span} keep={keep_bins}",
                )
            start_bin = max(0, min(req_start_bin, total_bins - keep_bins))
            if start_bin + keep_bins < req_end_bin:
                start_bin = req_end_bin - keep_bins
            start_bin = max(0, min(start_bin, total_bins - keep_bins))

        end_bin = start_bin + keep_bins
        chunk = np.array(arr[:, start_bin:end_bin], dtype=np.float32)
        base, chunk_start, _ = parse_chunk(path.stem)
        offset_ms = int(chunk_start or 0)
        start_ms = offset_ms + timebins_to_ms(start_bin)
        end_ms = offset_ms + timebins_to_ms(end_bin)
        out_name = f"{base}__ms_{start_ms}_{end_ms}.npy"
        out_path = out_dir / out_name
        # Keep deterministic parseable names. Collision indicates logic/cleanup bug.
        if out_path.exists() and out_path != path:
            fail("chunk_name_collision", status=3, message=f"output_exists={out_path}")
        if path == out_path:
            path.unlink()
            np.save(out_path, chunk)
            return out_path
        np.save(out_path, chunk)
        path.unlink()
        return out_path

    def build_split_meta(split_dir: Path):
        metas = []
        for p in sorted(split_dir.glob("*.npy")):
            arr = np.load(p, mmap_mode="r")
            t = int(arr.shape[1])
            u_int = file_unit_intervals(p, t)
            metas.append(
                {
                    "path": p,
                    "timebins": t,
                    "unit_intervals": u_int,
                    "units": set(u_int.keys()),
                }
            )
        return metas

    def support_counts(metas):
        counts = defaultdict(int)
        for m in metas:
            for uid in m["units"]:
                counts[uid] += 1
        return counts

    def required_window_for_meta(meta, counts):
        intervals = []
        for uid in meta["units"]:
            if counts.get(uid, 0) == 1:
                intervals.extend(meta["unit_intervals"].get(uid, []))
        if not intervals:
            return 0, None, None
        req_start = min(i[0] for i in intervals)
        req_end = max(i[1] for i in intervals)
        req_bins = max(1, ms_to_timebins(req_end - req_start))
        req_bins = min(req_bins, meta["timebins"])
        return req_bins, req_start, req_end

    def split_total_bins(split_dir: Path):
        total = 0
        for p in split_dir.glob("*.npy"):
            arr = np.load(p, mmap_mode="r")
            total += int(arr.shape[1])
        return total

    def reduce_split_to_target(split_dir: Path, target_bins: int):
        for _ in range(10000):
            cur = split_total_bins(split_dir)
            if cur <= target_bins:
                return cur
            excess = cur - target_bins
            metas = build_split_meta(split_dir)
            counts = support_counts(metas)
            candidates = []
            for m in metas:
                req_bins, req_start, req_end = required_window_for_meta(m, counts)
                removable = m["timebins"] - req_bins
                if removable > 0:
                    candidates.append((m, removable, req_start, req_end))
            if not candidates:
                return cur
            exact = [c for c in candidates if c[1] >= excess]
            if exact:
                m, removable, req_start, req_end = sorted(exact, key=lambda x: x[1])[0]
            else:
                m, removable, req_start, req_end = max(candidates, key=lambda x: x[1])
            remove_bins = min(excess, removable)
            keep_bins = m["timebins"] - remove_bins
            if keep_bins <= 0:
                move_to_dir(m["path"], pool_dir)
            else:
                crop_file_to_bins(
                    m["path"],
                    keep_bins,
                    split_dir,
                    req_start_ms=req_start,
                    req_end_ms=req_end,
                )
        fail("reduce_iter_limit", status=3)

    def increase_split_to_target(split_dir: Path, target_bins: int):
        for _ in range(10000):
            cur = split_total_bins(split_dir)
            if cur >= target_bins:
                return cur
            deficit = target_bins - cur
            pool_files = sorted(pool_dir.glob("*.npy"))
            if not pool_files:
                return cur
            candidates = []
            for p in pool_files:
                arr = np.load(p, mmap_mode="r")
                bins = int(arr.shape[1])
                if bins > 0:
                    candidates.append((p, bins))
            if not candidates:
                return cur
            whole = [c for c in candidates if c[1] <= deficit]
            if whole:
                p, _ = max(whole, key=lambda x: x[1])
                move_to_dir(p, split_dir)
                continue
            p, _ = min(candidates, key=lambda x: x[1])
            keep_bins = int(deficit)
            if keep_bins <= 0:
                return cur
            crop_file_to_bins(p, keep_bins, split_dir)
        fail("increase_iter_limit", status=3)

    def split_units(split_dir: Path):
        metas = build_split_meta(split_dir)
        out = set()
        for m in metas:
            out.update(m["units"])
        return out

    # Coverage-aware top-off.
    reduce_split_to_target(train_dir, train_target_bins)
    reduce_split_to_target(test_dir, test_target_bins)
    train_final_bins = increase_split_to_target(train_dir, train_target_bins)
    test_final_bins = increase_split_to_target(test_dir, test_target_bins)

    if train_final_bins != train_target_bins:
        fail("train_topoff_failed", status=3, message=f"{train_final_bins=} {train_target_bins=}")
    if test_final_bins != test_target_bins:
        fail("test_topoff_failed", status=3, message=f"{test_final_bins=} {test_target_bins=}")

    test_units = split_units(test_dir)
    train_units = split_units(train_dir)
    missing_test = sorted(all_units - test_units)
    missing_train = sorted(all_units - train_units)
    if missing_test or missing_train:
        fail(
            "coverage_lost_after_topoff",
            status=3,
            message=f"missing_test={missing_test} missing_train={missing_train}",
        )

    test_seconds = test_final_bins * hop / sr
    train_seconds_final = train_final_bins * hop / sr

    update_info(
        {
            "solver_feasible": True,
            "solver_status": 0,
            "solver_message": "ok",
            "failure_reason": "",
            "seed": int(seed) if seed is not None else None,
            "targets": {
                "test_bins": int(test_target_bins),
                "train_bins": int(train_target_bins),
                "test_seconds": float(test_target_bins * hop / sr),
                "train_seconds": float(train_target_bins * hop / sr),
            },
            "achieved": {
                "test_bins": int(test_final_bins),
                "train_bins": int(train_final_bins),
                "test_seconds": float(test_seconds),
                "train_seconds": float(train_seconds_final),
                "test_files": int(len(list(test_dir.glob("*.npy")))),
                "train_files": int(len(list(train_dir.glob("*.npy")))),
                "pool_left_files": int(len(list(pool_dir.glob("*.npy")))),
            },
        }
    )

    print(
        f"solver split complete: test_files={len(list(test_dir.glob('*.npy')))} ({test_seconds:.3f}s), "
        f"train_files={len(list(train_dir.glob('*.npy')))} ({train_seconds_final:.3f}s), "
        f"pool_left={len(list(pool_dir.glob('*.npy')))}"
    )
    print(
        f"targets: test={test_target_bins * hop / sr:.3f}s, "
        f"train={train_target_bins * hop / sr:.3f}s"
    )


def main():
    p = argparse.ArgumentParser(
        description="Solver-based disjoint train/test split with coverage-aware top-off."
    )
    p.add_argument("--pool_dir", required=True)
    p.add_argument("--annotation_json", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--train_dir", required=True)
    p.add_argument(
        "--train_seconds",
        required=True,
        help="Target train duration in seconds, or MAX to use all non-test pool duration.",
    )
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feasibility_json", default=None)
    args = p.parse_args()

    try:
        solve_and_split(
            pool_dir=Path(args.pool_dir),
            annotation_file=Path(args.annotation_json),
            test_dir=Path(args.test_dir),
            train_dir=Path(args.train_dir),
            train_seconds=args.train_seconds,
            test_ratio=float(args.test_ratio),
            seed=int(args.seed),
            feasibility_json=Path(args.feasibility_json) if args.feasibility_json else None,
        )
    except Exception as exc:
        print(f"solver split failed: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
