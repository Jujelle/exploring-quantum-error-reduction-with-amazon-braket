#!/usr/bin/env python3
"""Create threshold-sweep timing reports and charts from replica JSON files."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import time
from collections import defaultdict
from pathlib import Path

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


def load_records(results_dir: str | Path, prefix: str = "threshold") -> tuple[list[Path], list[dict]]:
    paths = sorted(Path(p) for p in glob.glob(str(Path(results_dir) / f"{prefix}*.json")))
    if not paths:
        raise FileNotFoundError(f"No result files found: {results_dir}/{prefix}*.json")

    records = []
    for path in paths:
        with open(path) as f:
            records.append(json.load(f))
    return paths, records


def aggregate_rows(records: list[dict]) -> list[dict]:
    acc = defaultdict(lambda: {"errors": 0, "shots": 0, "replicas": 0, "elapsed": 0.0})
    meta = {}
    for record in records:
        p = record.get("physical_error_rate", record.get("error_rate"))
        key = (record["distance"], round(float(p), 8))
        acc[key]["errors"] += record["logical_errors"]
        acc[key]["shots"] += record["shots"]
        acc[key]["replicas"] += 1
        acc[key]["elapsed"] += float(record.get("elapsed_sec", 0.0))
        meta[key] = record

    rows = []
    for key in sorted(acc):
        d, p = key
        item = acc[key]
        record = meta[key]
        rate, lo, hi = wilson_ci(item["errors"], item["shots"])
        rows.append({
            "distance": d,
            "physical_error_rate": p,
            "active_qubits": record["active_qubits"],
            "detectors": record["detectors"],
            "matching_edges": record["matching_edges"],
            "logical_errors": item["errors"],
            "shots": item["shots"],
            "replicas": item["replicas"],
            "logical_error_rate": rate,
            "ci95_low": lo,
            "ci95_high": hi,
            "total_elapsed_sec": item["elapsed"],
        })
    return rows


def timing_by_distance(records: list[dict]) -> list[dict]:
    acc = defaultdict(lambda: {"tasks": 0, "shots": 0, "elapsed": [], "subproc": []})
    for record in records:
        item = acc[record["distance"]]
        item["tasks"] += 1
        item["shots"] += int(record.get("shots", 0))
        elapsed = float(record.get("elapsed_sec", 0.0))
        item["elapsed"].append(elapsed)
        item["subproc"].append(float(record.get("subprocess_wall_sec", elapsed)))

    out = []
    for distance, item in sorted(acc.items()):
        elapsed = item["elapsed"]
        subproc = item["subproc"]
        out.append({
            "distance": distance,
            "tasks": item["tasks"],
            "shots": item["shots"],
            "sum_replica_elapsed_sec": sum(elapsed),
            "mean_replica_elapsed_sec": sum(elapsed) / len(elapsed) if elapsed else 0.0,
            "min_replica_elapsed_sec": min(elapsed) if elapsed else 0.0,
            "max_replica_elapsed_sec": max(elapsed) if elapsed else 0.0,
            "sum_subprocess_wall_sec": sum(subproc),
        })
    return out


def write_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_threshold(rows: list[dict], path: str | Path, title: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for d in sorted({row["distance"] for row in rows}):
        pts = sorted(
            [row for row in rows if row["distance"] == d],
            key=lambda row: row["physical_error_rate"],
        )
        ps = np.array([row["physical_error_rate"] for row in pts])
        rates = np.array([row["logical_error_rate"] for row in pts])
        shots = np.array([row["shots"] for row in pts])
        plot_rates = np.maximum(rates, 0.5 / shots)
        lows = np.array([row["ci95_low"] for row in pts])
        highs = np.array([row["ci95_high"] for row in pts])
        yerr = np.array([
            np.maximum(0.0, plot_rates - np.maximum(lows, 0.5 / shots)),
            np.maximum(0.0, highs - plot_rates),
        ])
        ax.errorbar(ps, plot_rates, yerr=yerr, marker="o", capsize=3, linewidth=1.8, label=f"d={d}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("logical error rate")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="code distance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def format_report(
    *,
    label: str,
    result_files: list[Path],
    rows: list[dict],
    wall_clock_total_sec: float,
    sum_replica_elapsed_sec: float,
    aggregate_elapsed_sec: float,
    distance_rows: list[dict],
    distance_timing_path: Path,
    wall_clock_note: str | None = None,
) -> str:
    lines = []
    lines.append(f"=== {label} timing report ===")
    if wall_clock_note:
        lines.append(f"NOTE: {wall_clock_note}")
        lines.append("")
    lines.append(f"Aggregated {len(result_files)} result files")
    lines.append(
        f"{'d':>3} {'p':>10} {'active':>7} {'det':>6} {'edges':>7} "
        f"{'errors':>9} {'shots':>10} {'replicas':>8} {'L_err':>10} "
        f"{'95% CI':>23} {'elapsed':>10}"
    )
    lines.append("-" * 124)
    for row in rows:
        lines.append(
            f"{row['distance']:>3} {row['physical_error_rate']:>10.6f} "
            f"{row['active_qubits']:>7} {row['detectors']:>6} {row['matching_edges']:>7} "
            f"{row['logical_errors']:>9} {row['shots']:>10} {row['replicas']:>8} "
            f"{row['logical_error_rate']:>9.4%} "
            f"[{row['ci95_low']:.4%}, {row['ci95_high']:.4%}] "
            f"{row['total_elapsed_sec']:>9.2f}s"
        )

    lines.append("")
    lines.append("Total calculation time")
    lines.append(f"wall_clock_total_sec: {wall_clock_total_sec:.3f}")
    lines.append(f"sum_replica_elapsed_sec: {sum_replica_elapsed_sec:.3f}")
    lines.append(f"aggregate_elapsed_sec: {aggregate_elapsed_sec:.3f}")

    lines.append("")
    lines.append("Calculation time by distance")
    lines.append(
        f"{'d':>3} {'tasks':>6} {'shots':>10} {'sum_replica':>13} "
        f"{'mean':>10} {'min':>10} {'max':>10} {'sum_subproc':>13}"
    )
    lines.append("-" * 92)
    for item in distance_rows:
        lines.append(
            f"{item['distance']:>3} {item['tasks']:>6} {item['shots']:>10} "
            f"{item['sum_replica_elapsed_sec']:>12.2f}s "
            f"{item['mean_replica_elapsed_sec']:>9.2f}s "
            f"{item['min_replica_elapsed_sec']:>9.2f}s "
            f"{item['max_replica_elapsed_sec']:>9.2f}s "
            f"{item['sum_subprocess_wall_sec']:>12.2f}s"
        )
    lines.append(f"\nSaved distance timing CSV: {distance_timing_path}")
    return "\n".join(lines)


def create_report(
    results_dir: str | Path,
    output_dir: str | Path,
    label: str,
    prefix: str = "threshold",
    wall_clock_total_sec: float | None = None,
    wall_clock_note: str | None = None,
) -> dict:
    aggregate_t0 = time.perf_counter()
    result_files, records = load_records(results_dir, prefix=prefix)
    rows = aggregate_rows(records)
    distance_rows = timing_by_distance(records)
    sum_replica_elapsed_sec = sum(float(record.get("elapsed_sec", 0.0)) for record in records)
    if wall_clock_total_sec is None:
        wall_clock_total_sec = sum_replica_elapsed_sec
        wall_clock_note = wall_clock_note or "wall_clock_total_sec is a placeholder equal to sum_replica_elapsed_sec."

    output_dir = Path(output_dir)
    summary_csv = write_csv(rows, output_dir / f"{label}_threshold_summary.csv")
    distance_csv = write_csv(distance_rows, output_dir / f"{label}_distance_timing.csv")
    chart_path = plot_threshold(rows, output_dir / f"{label}_threshold_curve.png", f"{label.upper()} surface-code threshold sweep")
    aggregate_elapsed_sec = time.perf_counter() - aggregate_t0
    report = format_report(
        label=label,
        result_files=result_files,
        rows=rows,
        wall_clock_total_sec=wall_clock_total_sec,
        sum_replica_elapsed_sec=sum_replica_elapsed_sec,
        aggregate_elapsed_sec=aggregate_elapsed_sec,
        distance_rows=distance_rows,
        distance_timing_path=distance_csv,
        wall_clock_note=wall_clock_note,
    )
    report_path = output_dir / f"{label}_timing_report.txt"
    report_path.write_text(report + "\n")
    return {
        "label": label,
        "tasks": len(result_files),
        "wall_clock_total_sec": wall_clock_total_sec,
        "sum_replica_elapsed_sec": sum_replica_elapsed_sec,
        "aggregate_elapsed_sec": aggregate_elapsed_sec,
        "summary_csv": str(summary_csv),
        "distance_csv": str(distance_csv),
        "chart_path": str(chart_path),
        "report_path": str(report_path),
        "report": report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", default="timing_reports")
    parser.add_argument("--label", required=True)
    parser.add_argument("--prefix", default="threshold")
    parser.add_argument("--wall-clock-total-sec", type=float, default=None)
    parser.add_argument("--wall-clock-note", default=None)
    args = parser.parse_args()

    result = create_report(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        label=args.label,
        prefix=args.prefix,
        wall_clock_total_sec=args.wall_clock_total_sec,
        wall_clock_note=args.wall_clock_note,
    )
    print(result["report"])
    print(f"Saved summary CSV: {result['summary_csv']}")
    print(f"Saved chart: {result['chart_path']}")
    print(f"Saved report: {result['report_path']}")


if __name__ == "__main__":
    main()
