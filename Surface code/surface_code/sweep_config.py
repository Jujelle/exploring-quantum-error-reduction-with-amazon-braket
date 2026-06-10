#!/usr/bin/env python3
"""Threshold sweep configuration for AWS PCS Slurm job arrays.

Each Slurm array task runs one independent Monte Carlo replica for one
(distance, physical error rate) point.

Default sweep:
    distances = [3, 5, 7, 9]
    p values  = 8 points from 0.001 to 0.02
    replicas  = 4 per (d, p)
    shots     = 100,000 per replica

Total tasks:
    4 distances * 8 p-values * 4 replicas = 128 tasks
"""
import argparse
import sys

import numpy as np

DISTANCES = [3, 5, 7, 9]
P_VALUES = list(np.geomspace(0.001, 0.02, 8))
REPLICAS_PER_POINT = 4
SHOTS_PER_REPLICA = 100_000


def grid():
    out = []
    for d in DISTANCES:
        for p in P_VALUES:
            for replica in range(REPLICAS_PER_POINT):
                out.append((d, float(p), replica))
    return out


def task_params(idx: int):
    g = grid()
    if not 0 <= idx < len(g):
        raise IndexError(f"task index {idx} out of range [0, {len(g)})")
    d, p, replica = g[idx]
    seed = (idx * 1_000_003 + 54321) % (2**31 - 1)
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
            print(f"{i:>4}  d={d:>2}  p={p:.6f}  replica={replica}  seed={seed}")
        return
    if args.idx is None:
        sys.exit("supply an index, or use --total / --shots / --table")

    d, p, replica, seed = task_params(args.idx)
    print(f"{d} {p:.8f} {replica} {seed}")


if __name__ == "__main__":
    main()
