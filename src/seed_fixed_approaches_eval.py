#!/usr/bin/env python3
"""
seed_fixed_approaches_eval.py

Evaluate fixed-context 'approaches' (ordered absolute indices) for each target chirp t
under the TinyBird model and cache per-target losses. Optionally run a genetic algorithm
(GA) over TEAMS of approach columns using cached losses only.

Correctness highlights
- Uses the authoritative TinyBird loss path EXACTLY (see eval_loss_for_indices).
- Works on a single spectrogram file selected by --index.
- Indices builder strictly matches spec (order kept, duplicates allowed, future context allowed).
- Robust caching: NPZ (losses), JSON (meta), CSV (best per-target).
- Idempotent resume: no recompute of finite cells; appends new approach columns without reordering.

Reproducibility
- Seeded RNGs (numpy, random, torch).
- Fingerprint ties cache to model_path, checkpoint, file_name, slots.

Resumability
- Loads existing cache; extends columns if new approaches appear.
- Periodic save via --save_every and again on completion.

Usage (examples)
- Auto-generate A=N_targets approaches and run GA teams of size 8:
  uv run src/seed_fixed_approaches_eval.py --model_path RUN_DIR --index 0 --device cuda --slots 10

- Supply approaches JSON:
  uv run src/seed_fixed_approaches_eval.py --model_path RUN_DIR --index 0 --approaches_json /path/approaches.json

- Quick dry-run on 128 random targets:
  uv run src/seed_fixed_approaches_eval.py --model_path RUN_DIR --index 0 --sample_targets 128

Outputs in --cache_dir (default ./ga_cache):
- seed_eval_<fp>.npz               # float32 losses [N_targets, A]
- seed_eval_<fp>.json              # meta (approaches list kept in column order)
- seed_eval_<fp>_best_per_target.csv
- seed_eval_<fp>_ga_summary.json   # best team from GA (if enabled)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Mirror your analyze_model.py imports/names exactly
from utils import load_model_from_checkpoint
from data_loader import SpectogramDataset  # note: "SpectogramDataset" (matches your file)


# =========================
# Indices Builder (SPEC)
# =========================
def build_indices_for_target_abs_fixed_order(
    t: int, approach: List[int], slots: int, n_valid: Optional[int] = None
) -> List[int]:
    """
    Return: [approach_truncated, predecessors, t]
      - Keep order; allow duplicates.
      - Do not drop entries >= t (future context is allowed).
      - If n_valid is not None: drop out-of-range (<0 or >= n_valid) *before* truncation.
      - Truncate approach to at most `slots`.
      - Autofill remaining slots with the immediate predecessors of t in ascending order
        (older → newer): [..., t-3, t-2, t-1], stopping at 0.
      - If fewer predecessors exist than needed, use what exists.
      - Append `t` as the final element.
    """
    if n_valid is not None:
        kept = [a for a in approach if 0 <= int(a) < int(n_valid)]
    else:
        kept = [int(a) for a in approach]

    kept = kept[: max(0, int(slots))]  # truncate to slots

    remaining = max(0, int(slots) - len(kept))
    if remaining > 0:
        # predecessors needed in ascending order (older→newer)
        # immediate predecessors (t-remaining ... t-1), clipped at 0, then keep those >=0
        preds = list(range(max(0, t - remaining), t))
        # preds is already ascending; if t<remaining this shortens naturally
        indices = kept + preds + [t]
    else:
        indices = kept + [t]

    return indices


# =========================
# Authoritative loss eval
# =========================
def eval_loss_for_indices(model, x, x_i, N, indices: Sequence[int]) -> float:
    """
    Run the authoritative TinyBird loss path EXACTLY and return Python float32.
    On CUDA OOM or messages containing 'out of memory'/'cuda', clear cache and return NaN.
    """
    try:
        # authoritative path (mirrors analyze_model.py)
        xs, x_is = model.sample_data_indices(x.clone(), x_i.clone(), N.clone(), list(indices))
        mblock = [len(indices) - 1]  # last element is the masked target
        h, idx_restore, bool_mask, bool_pad, T = model.forward_encoder(xs, x_is, mblock=mblock)
        pred = model.forward_decoder(h, idx_restore, T, bool_pad=bool_pad, attend_to_padded=False)
        loss = model.loss_mse(xs, pred, bool_mask)  # scalar tensor
        val = float(loss.detach().item())
        return np.float32(val).item()
    except RuntimeError as e:
        msg = str(e).lower()
        if ("out of memory" in msg or "cuda" in msg) and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return np.float32(np.nan).item()
        else:
            # propagate unknown runtime errors (so we notice actual bugs)
            raise


# =========================
# Auto-approach generation
# =========================
def _dense_windows(n_valid: int, w: int = 8) -> List[List[int]]:
    out = []
    if n_valid <= 0:
        return out
    for start in range(0, max(1, n_valid - w + 1), max(1, w // 2)):
        out.append(list(range(start, min(n_valid, start + w))))
    return out


def _evenly_spaced(n_valid: int, stride: int) -> List[List[int]]:
    return [list(range(0, n_valid, stride))]


def auto_generate_approaches(n_valid: int, num_auto: int, max_len: int, seed: int) -> List[List[int]]:
    """
    Mix of:
      - [] baseline
      - dense windows in early/mid/late regions (w≈8)
      - evenly spaced anchors (strides 4/8/16/32)
      - random lists (length U(0,max_len), entries U(0,n_valid-1)), duplicates allowed, arbitrary order
    Keep order; do NOT dedupe.
    """
    rng = random.Random(seed)
    out: List[List[int]] = []

    if n_valid > 0:
        out.append([])  # empty baseline

        # dense windows covering the sequence
        out.extend(_dense_windows(n_valid, w=min(8, max(1, n_valid))))

        # even anchors
        for s in [4, 8, 16, 32]:
            if s <= n_valid:
                out.extend(_evenly_spaced(n_valid, s))

        # randoms
        target_count = max(0, num_auto - len(out))
        for _ in range(target_count):
            L = rng.randint(0, max(0, max_len))
            if L == 0:
                out.append([])
            else:
                # duplicates allowed
                out.append([rng.randrange(0, n_valid) for _ in range(L)])

    # Truncate to requested count but keep diversity ordering
    if len(out) > num_auto:
        rng.shuffle(out)  # mild shuffle to avoid bias
        out = out[:num_auto]

    # Ensure lists of ints
    out = [[int(v) for v in a] for a in out]
    return out


# =========================
# Cache helpers
# =========================
def model_run_fingerprint(model_path: str, checkpoint: Optional[str], file_name: str, slots: int) -> str:
    s = f"mp:{model_path}|ckpt:{checkpoint or ''}|file:{file_name}|slots:{int(slots)}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def cache_paths(cache_dir: str, fp: str) -> Tuple[str, str, str, str]:
    npz_path = os.path.join(cache_dir, f"seed_eval_{fp}.npz")
    meta_json_path = os.path.join(cache_dir, f"seed_eval_{fp}.json")
    csv_path = os.path.join(cache_dir, f"seed_eval_{fp}_best_per_target.csv")
    ga_json_path = os.path.join(cache_dir, f"seed_eval_{fp}_ga_summary.json")
    return npz_path, meta_json_path, csv_path, ga_json_path


def _json_load(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _json_dump(path: str, obj) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_or_init_cache(
    npz_path: str, meta_path: str, num_targets: int, approaches: List[List[int]], meta_stub: dict
) -> Tuple[np.ndarray, dict, bool]:
    """
    Returns (losses [N_targets, A], meta, existed)
    - If cache exists: extend columns if new approaches are present (append NaN columns).
    - If not: initialize from scratch.
    """
    A_new = len(approaches)
    if os.path.exists(npz_path) and os.path.exists(meta_path):
        meta = _json_load(meta_path)
        existed = True
        # existing approaches (column order)
        old_approaches: List[List[int]] = meta.get("approaches", [])

        # map approach JSON to column index
        def keyify(ap):
            return json.dumps(ap, sort_keys=False)

        old_keys = [keyify(a) for a in old_approaches]
        new_keys = [keyify(a) for a in approaches]

        npz_broken = False
        z = None
        try:
            z = np.load(npz_path)
        except Exception:
            npz_broken = True

        losses = None
        if not npz_broken:
            try:
                if "losses" not in z or not isinstance(z["losses"], np.ndarray):
                    npz_broken = True
                else:
                    arr = z["losses"]
                    if arr.dtype != np.float32 or arr.ndim != 2:
                        npz_broken = True
                    else:
                        losses = arr.astype(np.float32)
            except Exception:
                npz_broken = True
            finally:
                if z is not None:
                    try:
                        z.close()
                    except Exception:
                        pass

        if npz_broken:
            # Rename broken file to .bad (ignore errors)
            try:
                bad_path = npz_path + ".bad"
                os.rename(npz_path, bad_path)
            except Exception:
                pass
            # Use meta's approaches if present, else incoming approaches
            approaches_for_cols = meta.get("approaches", approaches)
            A_existing = len(approaches_for_cols)
            losses = np.full((num_targets, A_existing), np.nan, dtype=np.float32)
            meta["approaches"] = [list(map(int, a)) for a in approaches_for_cols]
            print(f"[WARN] Cache NPZ appears corrupted; reinitialized at {npz_path}")
            return losses, meta, existed

        # Append columns for any truly new approaches (preserving old column order)
        append_cols = []
        append_approaches = []
        for k, ap in zip(new_keys, approaches):
            if k not in old_keys:
                append_cols.append(k)
                append_approaches.append(ap)

        if append_cols:
            extra = np.full((num_targets, len(append_cols)), np.nan, dtype=np.float32)
            losses = np.concatenate([losses, extra], axis=1)
            meta["approaches"].extend(append_approaches)

        # If N_targets changed across runs (shouldn't, but be defensive), resize rows
        if losses.shape[0] != num_targets:
            # expand or crop with NaNs
            new_losses = np.full((num_targets, losses.shape[1]), np.nan, dtype=np.float32)
            rows = min(num_targets, losses.shape[0])
            new_losses[:rows, :] = losses[:rows, :]
            losses = new_losses

        return losses, meta, existed
    else:
        existed = False
        losses = np.full((num_targets, A_new), np.nan, dtype=np.float32)
        meta = dict(meta_stub)
        meta["approaches"] = [list(map(int, a)) for a in approaches]
        return losses, meta, existed


def save_cache(npz_path: str, meta_path: str, losses: np.ndarray, meta: dict) -> None:
    # Ensure directory exists
    dirpath = os.path.dirname(npz_path) or "."
    os.makedirs(dirpath, exist_ok=True)

    # Write NPZ atomically via a pre-created named temp file in the same directory
    import tempfile
    with tempfile.NamedTemporaryFile(
        prefix=os.path.basename(npz_path) + ".",
        suffix=".tmp",
        dir=dirpath,
        delete=False,
    ) as tmpf:
        tmp_npz = tmpf.name

    try:
        np.savez_compressed(tmp_npz, losses=losses.astype(np.float32, copy=False))
        # Atomic replace on same filesystem
        os.replace(tmp_npz, npz_path)
    except Exception:
        # Best-effort cleanup of the temp file if anything goes wrong
        try:
            if os.path.exists(tmp_npz):
                os.remove(tmp_npz)
        finally:
            raise

    # Write JSON meta (uses its own atomic writer)
    _json_dump(meta_path, meta)


def write_best_csv(csv_path: str, losses: np.ndarray, meta: dict) -> None:
    """
    Write rows: target_idx,best_approach_col,loss,best_approach_json
    Overwrites each time (cheap).
    """
    approaches = meta.get("approaches", [])
    A = len(approaches)
    with open(csv_path, "w") as f:
        f.write("target_idx,best_approach_col,loss,best_approach_json\n")
        for t in range(losses.shape[0]):
            col = -1
            best = np.inf
            for j in range(A):
                v = losses[t, j]
                if np.isfinite(v) and v < best:
                    best = v
                    col = j
            if col < 0:
                f.write(f"{t},-1,NaN,[]\n")
            else:
                ap_json = json.dumps(approaches[col], separators=(",", ":"))
                f.write(f"{t},{col},{best:.7g},{ap_json}\n")


# =========================
# GA over teams (cached)
# =========================
@dataclass
class TeamEval:
    fitness: float  # mean finite best loss across targets covered by the team
    coverage: float  # fraction of targets with at least one finite team loss
    covered_count: int
    considered_count: int
    best_cols_per_target: Optional[np.ndarray]  # argmin col per target among the team (or -1)


def _evaluate_team(losses: np.ndarray, team_cols: Sequence[int]) -> TeamEval:
    """
    Compute team fitness using cached losses ONLY.
    fitness = mean over targets with ≥1 finite loss among the team.
    coverage = covered_targets / total_targets.
    """
    if len(team_cols) == 0:
        return TeamEval(
            fitness=np.inf, coverage=0.0, covered_count=0, considered_count=losses.shape[0], best_cols_per_target=None
        )

    L = losses[:, list(team_cols)]  # [T, K]
    finite_mask = np.isfinite(L)
    covered = finite_mask.any(axis=1)  # [T]
    covered_idx = np.where(covered)[0]
    if covered_idx.size == 0:
        return TeamEval(
            fitness=np.inf, coverage=0.0, covered_count=0, considered_count=losses.shape[0], best_cols_per_target=None
        )

    # Replace NaNs with +inf to enable argmin
    L2 = L.copy()
    L2[~finite_mask] = np.inf
    best_vals = np.min(L2, axis=1)  # [T]
    best_cols = np.argmin(L2, axis=1)  # [T]
    # Compute fitness over covered targets only
    fit = float(np.mean(best_vals[covered_idx]))
    cov = float(covered_idx.size / losses.shape[0])
    # For targets not covered, set best col = -1
    best_cols_full = np.full(losses.shape[0], -1, dtype=np.int32)
    best_cols_full[covered] = np.array(team_cols, dtype=np.int32)[best_cols[covered]]

    return TeamEval(
        fitness=fit,
        coverage=cov,
        covered_count=int(covered_idx.size),
        considered_count=int(losses.shape[0]),
        best_cols_per_target=best_cols_full,
    )


def _seed_population(
    losses: np.ndarray, approaches: List[List[int]], K: int, P: int, rng: random.Random
) -> List[List[int]]:
    A = len(approaches)
    if A == 0:
        return [[] for _ in range(P)]
    # rank columns by global mean finite loss
    col_scores = []
    for j in range(A):
        col = losses[:, j]
        m = np.nanmean(col)
        col_scores.append((np.inf if np.isnan(m) else m, j))
    col_scores.sort()
    top_cols = [j for _, j in col_scores[: max(1, min(A, K * 2))]]

    pop: List[List[int]] = []

    # 1) Some seeded by "good" columns
    for _ in range(min(P // 3, max(1, P // 3))):
        s = set()
        # prefer top columns, then fill randomly
        for j in rng.sample(top_cols, k=min(len(top_cols), K)):
            s.add(j)
            if len(s) >= K:
                break
        while len(s) < K:
            s.add(rng.randrange(0, A))
        pop.append(list(s)[:K])

    # 2) Some diverse (strides/dense/random) — just random subsets of columns to ensure diversity
    for _ in range(min(P // 3, max(1, P // 3))):
        s = set()
        while len(s) < K:
            s.add(rng.randrange(0, A))
        pop.append(list(s)[:K])

    # 3) Fill the rest purely random
    while len(pop) < P:
        s = set()
        while len(s) < K:
            s.add(rng.randrange(0, A))
        pop.append(list(s)[:K])

    return pop


def _crossover(parent1: List[int], parent2: List[int], K: int, rng: random.Random) -> List[int]:
    if not parent1 and not parent2:
        return []
    # union then downselect greedily by marginal coverage/benefit is expensive; use 1-point crossover then dedup
    cut1 = rng.randint(0, len(parent1)) if parent1 else 0
    cut2 = rng.randint(0, len(parent2)) if parent2 else 0
    child = parent1[:cut1] + parent2[cut2:]
    # dedup while preserving order, then clip to K
    seen = set()
    uniq = []
    for c in child:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
        if len(uniq) >= K:
            break
    # If too short, pad with random
    while len(uniq) < K and parent1:
        c = rng.choice(parent1)
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq[:K]


def _mutate(team: List[int], A: int, pm: float, rng: random.Random) -> List[int]:
    if rng.random() < pm and A > 0 and team:
        i = rng.randrange(0, len(team))
        # replace with a random column
        team = list(team)
        team[i] = rng.randrange(0, A)
        # de-duplicate while preserving order
        seen = set()
        out = []
        for c in team:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out[: len(team)]
    return team


def run_ga(
    losses: np.ndarray,
    approaches: List[List[int]],
    team_size: int,
    pop_size: int,
    gens: int,
    elite: int,
    mut_prob: float,
    seed: int,
    progress: bool = True,
) -> Tuple[List[int], TeamEval]:
    rng = random.Random(seed)
    A = len(approaches)
    pop = _seed_population(losses, approaches, team_size, pop_size, rng)

    def eval_team(cols: List[int]) -> TeamEval:
        return _evaluate_team(losses, cols)

    # initial evaluation
    scored = [(eval_team(t), t) for t in pop]

    # lower fitness is better; if equal, prefer higher coverage
    def better(a, b):
        ta, tb = a[0], b[0]
        if not math.isfinite(ta.fitness) and math.isfinite(tb.fitness):
            return False
        if math.isfinite(ta.fitness) and not math.isfinite(tb.fitness):
            return True
        if ta.fitness != tb.fitness:
            return ta.fitness < tb.fitness
        return ta.coverage > tb.coverage

    pbar = tqdm(range(gens), desc="GA generations", disable=not progress)
    for _ in pbar:
        # sort population
        scored.sort(key=lambda x: (math.inf if not math.isfinite(x[0].fitness) else x[0].fitness, -x[0].coverage))
        next_pop: List[List[int]] = [t for (_, t) in scored[:elite]]

        # fill the rest by tournaments + crossover
        while len(next_pop) < pop_size:
            a, b, c, d = rng.sample(scored, k=min(4, len(scored)))
            p1 = a if better(a, b) else b
            p2 = c if better(c, d) else d
            child = _crossover(p1[1], p2[1], team_size, rng)
            child = _mutate(child, A, mut_prob, rng)
            if child:
                next_pop.append(child)
            else:
                # fallback random if empty
                next_pop.append(_mutate([rng.randrange(0, A) for _ in range(team_size)], A, mut_prob, rng))

        pop = next_pop
        scored = [(eval_team(t), t) for t in pop]
        best = min(
            scored, key=lambda x: (math.inf if not math.isfinite(x[0].fitness) else x[0].fitness, -x[0].coverage)
        )
        pbar.set_postfix({"best_f": f"{best[0].fitness:.6g}", "cov": f"{best[0].coverage:.3f}"})

    best = min(scored, key=lambda x: (math.inf if not math.isfinite(x[0].fitness) else x[0].fitness, -x[0].coverage))
    return best[1], best[0]


# =========================
# CLI / Orchestration
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fixed-context approaches and run GA over teams.")
    p.add_argument("--model_path", type=str, required=True, help="Path to run dir (config.json + weights/).")
    p.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint file (optional).")
    p.add_argument("--data_dir", type=str, default=None, help="Path to .pt spectrograms (else val_dir from config).")
    p.add_argument("--index", type=int, default=0, help="Dataset index of the single file to analyze.")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for inference.",
    )
    p.add_argument("--slots", type=int, default=10, help="Number of context slots before appending t (≤10).")
    p.add_argument("--max_len", type=int, default=9, help="Max length for any approach (pre-truncation).")

    # Approaches source (one of)
    p.add_argument("--approaches_json", type=str, default=None, help="Path to JSON list-of-lists (approaches).")
    p.add_argument(
        "--num_auto",
        type=int,
        default=None,
        help="If not providing JSON, auto-generate this many approaches (default=N_targets).",
    )

    # Cache / control
    p.add_argument("--cache_dir", type=str, default="./ga_cache", help="Cache directory.")
    p.add_argument("--save_every", type=int, default=2000, help="Persist after this many evaluations.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument(
        "--sample_targets", type=int, default=None, help="Evaluate only a random subset of targets (quick dry runs)."
    )

    # GA
    p.add_argument("--ga_enable", action="store_true", default=True, help="Enable GA (default on).")
    p.add_argument("--ga_team_size", type=int, default=8, help="Team size K.")
    p.add_argument("--ga_pop", type=int, default=32, help="Population size.")
    p.add_argument("--ga_gens", type=int, default=50, help="Generations.")
    p.add_argument("--ga_elite", type=int, default=2, help="Elites kept each gen.")
    p.add_argument("--ga_mut_prob", type=float, default=0.25, help="Mutation probability per team.")
    return p.parse_args()


def main():
    args = parse_args()

    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load model
    print("=" * 60)
    print("TinyBird Fixed-Context Approaches Evaluation")
    print("=" * 60)
    print(f"Loading model: {args.model_path}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")

    model, config = load_model_from_checkpoint(
        run_dir=args.model_path, checkpoint_file=args.checkpoint, fallback_to_random=False
    )

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # Data dir
    if args.data_dir is not None:
        data_dir = args.data_dir
    elif "val_dir" in config:
        data_dir = config["val_dir"]
    else:
        raise ValueError("No data directory specified. Provide --data_dir or ensure val_dir in config.json")

    # Dataset + pick item
    dataset = SpectogramDataset(
        dir=data_dir, n_mels=config.get("mels", 128), n_timebins=config.get("num_timebins", 1024), pad_crop=True
    )
    if not (0 <= args.index < len(dataset)):
        raise ValueError(f"--index {args.index} out of range (dataset size={len(dataset)}).")
    x, x_i, x_l, N, file_name = dataset[args.index]

    # Batchify / device
    x = x.unsqueeze(0).float().to(device)
    x_i = x_i.unsqueeze(0).to(device)
    N = torch.tensor([int(N)], dtype=torch.long, device=device)

    # Mirror analyze_model.py: compactify once before all evals
    x, x_i = model.compactify_data(x.clone(), x_i.clone(), N.clone())

    N_targets = int(N.max().item())
    if N_targets <= 1:
        raise ValueError("Not enough valid chirps (N_targets <= 1). Need at least 2.")

    # Approaches
    if args.approaches_json:
        with open(args.approaches_json, "r") as f:
            approaches_raw = json.load(f)
        if not isinstance(approaches_raw, list) or any(not isinstance(a, list) for a in approaches_raw):
            raise ValueError("--approaches_json must be a JSON list of lists")
        approaches = [list(map(int, a))[: args.max_len] for a in approaches_raw]
    else:
        num_auto = args.num_auto if args.num_auto is not None else N_targets
        approaches = auto_generate_approaches(N_targets, num_auto=num_auto, max_len=args.max_len, seed=args.seed)

    # Cache
    fp = model_run_fingerprint(args.model_path, args.checkpoint, file_name, args.slots)
    npz_path, meta_path, csv_path, ga_json_path = cache_paths(args.cache_dir, fp)

    meta_stub = {
        "file_name": file_name,
        "num_targets": N_targets,
        "slots": int(args.slots),
        "model_path": args.model_path,
        "checkpoint": args.checkpoint,
        "device": str(device),
    }
    losses, meta, existed = load_or_init_cache(npz_path, meta_path, N_targets, approaches, meta_stub)
    # Column count reflects meta["approaches"]
    A = len(meta["approaches"])

    # Target set (skip t=0)
    targets_full = list(range(1, N_targets))
    if args.sample_targets is not None:
        k = max(1, min(len(targets_full), int(args.sample_targets)))
        rng = random.Random(args.seed)
        targets = sorted(rng.sample(targets_full, k))
        print(f"Sampling {k}/{len(targets_full)} targets for evaluation.")
    else:
        targets = targets_full

    # Coverage stat BEFORE eval
    cov_any = np.isfinite(losses[:, :A]).any(axis=1)
    coverage_rate = float(np.mean(cov_any)) if losses.shape[0] > 0 else 0.0
    print(
        f"Cache status: {('resumed' if existed else 'new')}, approaches={A}, "
        f"coverage (any finite): {coverage_rate:.3f}"
    )

    # Evaluation loop (respect cache: only fill NaNs)
    eval_counter = 0
    to_eval = 0
    for t in targets:
        # count pending cells for progress bar sizing
        row = losses[t, :]
        to_eval += int(np.sum(~np.isfinite(row)))

    if to_eval > 0:
        pbar = tqdm(total=to_eval, desc="Evaluating losses")
    else:
        pbar = None

    for t in targets:
        for col, approach in enumerate(meta["approaches"]):
            if np.isfinite(losses[t, col]):
                continue  # already done

            # Build indices per SPEC
            indices = build_indices_for_target_abs_fixed_order(
                t=int(t), approach=list(map(int, approach)), slots=int(args.slots), n_valid=N_targets
            )
            # Compute
            with torch.no_grad():
                v = eval_loss_for_indices(model, x, x_i, N, indices)
            losses[t, col] = np.float32(v)
            eval_counter += 1
            if pbar is not None:
                pbar.update(1)

            # periodic save
            if args.save_every > 0 and (eval_counter % args.save_every == 0):
                save_cache(npz_path, meta_path, losses, meta)
                write_best_csv(csv_path, losses, meta)

    if pbar is not None:
        pbar.close()

    # Final save
    save_cache(npz_path, meta_path, losses, meta)
    write_best_csv(csv_path, losses, meta)

    # Coverage after eval
    cov_any = np.isfinite(losses[:, :A]).any(axis=1)
    coverage_rate = float(np.mean(cov_any)) if losses.shape[0] > 0 else 0.0
    covered = int(np.sum(cov_any))
    print(f"Coverage after eval: {covered}/{losses.shape[0]} = {coverage_rate:.3f}")

    # GA (cached-only)
    if args.ga_enable and A > 0:
        best_team_cols, te = run_ga(
            losses=losses,
            approaches=meta["approaches"],
            team_size=int(args.ga_team_size),
            pop_size=int(args.ga_pop),
            gens=int(args.ga_gens),
            elite=int(args.ga_elite),
            mut_prob=float(args.ga_mut_prob),
            seed=int(args.seed),
            progress=True,
        )
        print("\nBest GA team")
        print(f"  columns: {best_team_cols}")
        print(f"  fitness (mean best loss over covered targets): {te.fitness:.6g}")
        print(f"  coverage: {te.coverage:.3f} ({te.covered_count}/{te.considered_count})")

        # Save GA summary
        ga_summary = {
            "best_team_cols": best_team_cols,
            "fitness": te.fitness,
            "coverage": te.coverage,
            "covered_count": te.covered_count,
            "considered_count": te.considered_count,
            "approaches_for_team": [meta["approaches"][c] for c in best_team_cols],
        }
        _json_dump(ga_json_path, ga_summary)
        print(f"Saved GA summary → {ga_json_path}")

    print("\nOutputs")
    print(f"  Loss NPZ : {npz_path}")
    print(f"  Meta JSON: {meta_path}")
    print(f"  Best CSV : {csv_path}")
    if args.ga_enable:
        print(f"  GA JSON  : {ga_json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
