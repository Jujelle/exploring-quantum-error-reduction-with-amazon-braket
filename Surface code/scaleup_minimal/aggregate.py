#!/usr/bin/env python3
"""Aggregate Slurm array JSON outputs into a scale-up table and plot."""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--prefix", default="scaleup")
    parser.add_argument("--csv", default="scaleup_summary.csv")
    parser.add_argument("--plot", default="scaleup_logical_error.png")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, f"{args.prefix}_*.json")))
    if not files:
        raise SystemExit(f"No result files found: {args.results_dir}/{args.prefix}_*.json")

    acc = defaultdict(lambda: {"errors": 0, "shots": 0, "replicas": 0, "elapsed": 0.0})
    meta = {}
    for path in files:
        with open(path) as f:
            r = json.load(f)
        key = (r["distance"], r["physical_error_rate"])
        acc[key]["errors"] += r["logical_errors"]
        acc[key]["shots"] += r["shots"]
        acc[key]["replicas"] += 1
        acc[key]["elapsed"] += r.get("elapsed_sec", 0.0)
        meta[key] = r

    rows = []
    print(f"Aggregated {len(files)} files")
    print(f"{'d':>3} {'p':>9} {'active':>7} {'det':>6} {'edges':>7} "
          f"{'errors':>9} {'shots':>10} {'replicas':>8} {'L_err':>10}")
    print("-" * 82)
    for key in sorted(acc):
        d, p = key
        a = acc[key]
        m = meta[key]
        rate = a["errors"] / a["shots"] if a["shots"] else 0.0
        # Binomial standard error. This is enough for a first scale-up demo.
        stderr = (rate * (1 - rate) / a["shots"]) ** 0.5 if a["shots"] else 0.0
        rows.append({
            "distance": d,
            "physical_error_rate": p,
            "active_qubits": m["active_qubits"],
            "detectors": m["detectors"],
            "matching_edges": m["matching_edges"],
            "logical_errors": a["errors"],
            "shots": a["shots"],
            "replicas": a["replicas"],
            "logical_error_rate": rate,
            "stderr": stderr,
        })
        print(f"{d:>3} {p:>9.6f} {m['active_qubits']:>7} {m['detectors']:>6} "
              f"{m['matching_edges']:>7} {a['errors']:>9} {a['shots']:>10} "
              f"{a['replicas']:>8} {rate:>9.4%}")

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {args.csv}")

    ds = [r["distance"] for r in rows]
    rates = [r["logical_error_rate"] for r in rows]
    yerr = [r["stderr"] for r in rows]
    labels = [
        f"{r['active_qubits']} qubits\n{r['detectors']} det."
        for r in rows
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.errorbar(ds, rates, yerr=yerr, marker="o", capsize=4, linewidth=1.8)
    for x, y, label in zip(ds, rates, labels):
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("code distance d")
    ax.set_ylabel("logical error rate")
    ax.set_title("Surface-code scale-up at p = 0.005")
    ax.set_xticks(ds)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()
