#!/usr/bin/env python3
"""Aggregate threshold-sweep JSON outputs into CSV and threshold plot."""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return p_hat, max(0.0, center - half), min(1.0, center + half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--prefix", default="threshold")
    parser.add_argument("--csv", default="threshold_summary.csv")
    parser.add_argument("--plot", default="threshold_curve.png")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, f"{args.prefix}_*.json")))
    if not files:
        raise SystemExit(f"No result files found: {args.results_dir}/{args.prefix}_*.json")

    acc = defaultdict(lambda: {"errors": 0, "shots": 0, "replicas": 0, "elapsed": 0.0})
    meta = {}
    for path in files:
        with open(path) as f:
            r = json.load(f)
        p = r.get("physical_error_rate", r.get("error_rate"))
        key = (r["distance"], round(float(p), 8))
        acc[key]["errors"] += r["logical_errors"]
        acc[key]["shots"] += r["shots"]
        acc[key]["replicas"] += 1
        acc[key]["elapsed"] += r.get("elapsed_sec", 0.0)
        meta[key] = r

    rows = []
    print(f"Aggregated {len(files)} files")
    print(f"{'d':>3} {'p':>10} {'active':>7} {'det':>6} {'edges':>7} "
          f"{'errors':>9} {'shots':>10} {'replicas':>8} {'L_err':>10} {'95% CI':>23}")
    print("-" * 110)
    for key in sorted(acc):
        d, p = key
        a = acc[key]
        m = meta[key]
        rate, lo, hi = wilson_ci(a["errors"], a["shots"])
        row = {
            "distance": d,
            "physical_error_rate": p,
            "active_qubits": m["active_qubits"],
            "detectors": m["detectors"],
            "matching_edges": m["matching_edges"],
            "logical_errors": a["errors"],
            "shots": a["shots"],
            "replicas": a["replicas"],
            "logical_error_rate": rate,
            "ci95_low": lo,
            "ci95_high": hi,
            "total_elapsed_sec": a["elapsed"],
        }
        rows.append(row)
        print(f"{d:>3} {p:>10.6f} {m['active_qubits']:>7} {m['detectors']:>6} "
              f"{m['matching_edges']:>7} {a['errors']:>9} {a['shots']:>10} "
              f"{a['replicas']:>8} {rate:>9.4%} [{lo:.4%}, {hi:.4%}]")

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {args.csv}")

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for d in sorted({r["distance"] for r in rows}):
        pts = sorted([r for r in rows if r["distance"] == d],
                     key=lambda r: r["physical_error_rate"])
        ps = np.array([r["physical_error_rate"] for r in pts])
        rates = np.array([r["logical_error_rate"] for r in pts])
        shots = np.array([r["shots"] for r in pts])
        # Log-scale plots cannot show exactly zero. For zero-error points,
        # place the marker at half an event and let the CI indicate the bound.
        plot_rates = np.maximum(rates, 0.5 / shots)
        lows = np.array([r["ci95_low"] for r in pts])
        highs = np.array([r["ci95_high"] for r in pts])
        yerr = np.array([
            np.maximum(0.0, plot_rates - np.maximum(lows, 0.5 / shots)),
            np.maximum(0.0, highs - plot_rates),
        ])
        ax.errorbar(ps, plot_rates, yerr=yerr, marker="o", capsize=3,
                    linewidth=1.8, label=f"d={d}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("logical error rate")
    ax.set_title("Surface-code threshold sweep on AWS PCS")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="code distance")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
