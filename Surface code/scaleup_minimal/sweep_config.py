#!/usr/bin/env python3
"""Small d=3/5/7/9 scale-up sweep for AWS PCS Slurm job arrays.

The Slurm array task id maps to one independent Monte Carlo replica:

    task id -> (distance, physical_error_rate, replica, seed)

Current sweep:
    distances = [3, 5, 7, 9]
    p = 0.005
    replicas per distance = 4
    shots per replica = 100,000

Total tasks:
    4 distances * 1 p * 4 replicas = 16
"""
import argparse
import sys

DISTANCES = [3, 5, 7, 9]
P_VALUES = [0.005]
REPLICAS_PER_POINT = 4
SHOTS_PER_REPLICA = 100_000


def grid():
    out = []
    for d in DISTANCES:
        for p in P_VALUES:
            for replica in range(REPLICAS_PER_POINT):
                out.append((d, p, replica))
    return out


def task_params(idx: int):
    g = grid()
    if not 0 <= idx < len(g):
        raise IndexError(f"task index {idx} out of range [0, {len(g)})")
    d, p, replica = g[idx]
    seed = (idx * 1_000_003 + 12345) % (2**31 - 1)
    return d, p, replica, seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", nargs="?", type=int)
    parser.add_argument("--total", action="store_true")
    parser.add_argument("--shots", action="store_true")
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args()

    if args.total:
        print(len(grid()))
        return
    if args.shots:
        print(SHOTS_PER_REPLICA)
        return
    if args.table:
        for i, _ in enumerate(grid()):
            d, p, replica, seed = task_params(i)
            print(f"{i:>3}  d={d:>2}  p={p:.6f}  replica={replica}  seed={seed}")
        return
    if args.idx is None:
        sys.exit("supply an index, or use --total / --shots / --table")

    d, p, replica, seed = task_params(args.idx)
    # Whitespace-separated so bash can read it with:
    # read -r DISTANCE ERROR_RATE REPLICA SEED < <(python3 sweep_config.py "$TASK_ID")
    print(f"{d} {p:.6f} {replica} {seed}")


if __name__ == "__main__":
    main()
