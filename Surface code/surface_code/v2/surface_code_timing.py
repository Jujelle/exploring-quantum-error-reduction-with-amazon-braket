#!/usr/bin/env python3
"""Timing helpers for surface-code EC2 and PCS threshold sweeps.

The heavy Monte Carlo work lives in ``surface_code_hpc.py``. This module
contains the small reusable timing layer used by notebooks and post-processing:
build task lists, run a small local/EC2 benchmark, summarize replica JSON files,
and compare EC2 versus PCS wall-clock time.
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from surface_code_hpc import run_replica


def describe_sweep(
    distances: Iterable[int],
    p_values: Iterable[float],
    replicas_per_point: int,
    shots_per_replica: int,
) -> dict:
    """Return basic size information for a threshold sweep."""
    distances = list(distances)
    p_values = list(p_values)
    task_count = len(distances) * len(p_values) * replicas_per_point
    return {
        "distances": distances,
        "p_points": len(p_values),
        "replicas_per_point": replicas_per_point,
        "shots_per_replica": shots_per_replica,
        "task_count": task_count,
        "total_shots": task_count * shots_per_replica,
    }


def make_tasks(
    distances: Iterable[int],
    p_values: Iterable[float],
    replicas_per_point: int,
    shots_per_replica: int,
    base_seed: int = 20260605,
) -> list[dict]:
    """Create independent replica tasks for a threshold sweep."""
    tasks = []
    idx = 0
    for distance in distances:
        for p in p_values:
            for replica in range(replicas_per_point):
                tasks.append({
                    "idx": idx,
                    "distance": int(distance),
                    "p": float(p),
                    "replica": int(replica),
                    "seed": int((base_seed + idx * 1_000_003) % (2**31 - 1)),
                    "shots": int(shots_per_replica),
                })
                idx += 1
    return tasks


def run_timed_task(task: dict) -> dict:
    """Run one replica and add notebook/EC2 wall time for that task."""
    t0 = time.perf_counter()
    record = run_replica(
        distance=task["distance"],
        rounds=task["distance"],
        p=task["p"],
        shots=task["shots"],
        seed=task["seed"],
    )
    return {**task, **record, "elapsed_sec": time.perf_counter() - t0}


def summarize_timing(
    records: Iterable[dict],
    wall_clock_total_sec: float,
    aggregate_elapsed_sec: float = 0.0,
    label: str = "run",
    max_workers: int | None = None,
    host: str | None = None,
) -> dict:
    """Summarize replica records into total and per-distance timing."""
    records = list(records)
    distance_acc = defaultdict(lambda: {
        "tasks": 0,
        "shots": 0,
        "logical_errors": 0,
        "elapsed": [],
    })

    for record in records:
        item = distance_acc[record["distance"]]
        item["tasks"] += 1
        item["shots"] += record["shots"]
        item["logical_errors"] += record["logical_errors"]
        item["elapsed"].append(float(record.get("elapsed_sec", 0.0)))

    timing_by_distance = []
    for distance, item in sorted(distance_acc.items()):
        elapsed = item["elapsed"]
        elapsed_sum = sum(elapsed)
        timing_by_distance.append({
            "distance": distance,
            "tasks": item["tasks"],
            "shots": item["shots"],
            "logical_errors": item["logical_errors"],
            "sum_replica_elapsed_sec": elapsed_sum,
            "mean_replica_elapsed_sec": elapsed_sum / len(elapsed) if elapsed else 0.0,
            "min_replica_elapsed_sec": min(elapsed) if elapsed else 0.0,
            "max_replica_elapsed_sec": max(elapsed) if elapsed else 0.0,
        })

    return {
        "label": label,
        "host": host or platform.node(),
        "max_workers": max_workers,
        "total_tasks": len(records),
        "wall_clock_total_sec": float(wall_clock_total_sec),
        "sum_replica_elapsed_sec": sum(float(r.get("elapsed_sec", 0.0)) for r in records),
        "aggregate_elapsed_sec": float(aggregate_elapsed_sec),
        "timing_by_distance": timing_by_distance,
    }


def run_parallel_benchmark(
    distances: Iterable[int],
    p_values: Iterable[float],
    replicas_per_point: int,
    shots_per_replica: int,
    max_workers: int,
    label: str = "ec2_notebook",
    base_seed: int = 20260605,
    results_dir: str | Path | None = None,
) -> tuple[list[dict], dict, Path | None]:
    """Run a small EC2/notebook benchmark and optionally save its timing JSON."""
    tasks = make_tasks(
        distances=distances,
        p_values=p_values,
        replicas_per_point=replicas_per_point,
        shots_per_replica=shots_per_replica,
        base_seed=base_seed,
    )

    sweep_t0 = time.perf_counter()
    records = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_timed_task, task) for task in tasks]
        for future in cf.as_completed(futures):
            records.append(future.result())
    wall_clock_total_sec = time.perf_counter() - sweep_t0

    aggregate_t0 = time.perf_counter()
    summary = summarize_timing(
        records=records,
        wall_clock_total_sec=wall_clock_total_sec,
        label=label,
        max_workers=max_workers,
    )
    summary["aggregate_elapsed_sec"] = time.perf_counter() - aggregate_t0

    summary_path = None
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        run_id = time.strftime("%Y%m%d_%H%M%S")
        summary_path = results_dir / f"run_summary_{label}_{run_id}.json"
        save_timing_summary(summary, summary_path)

    return records, summary, summary_path


def save_timing_summary(summary: dict, path: str | Path) -> Path:
    """Write a timing summary JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


def load_timing_summary(path: str | Path) -> dict:
    """Read a timing summary JSON file."""
    path = Path(path)
    with open(path) as f:
        summary = json.load(f)
    summary.setdefault("label", path.stem)
    return summary


def load_replica_records(result_glob: str | Path) -> list[dict]:
    """Load PCS/EC2 replica JSON records matching a glob pattern."""
    paths = sorted(Path(path) for path in glob.glob(str(result_glob)))
    if not paths:
        raise FileNotFoundError(f"No replica JSON files matched: {result_glob}")

    records = []
    for path in paths:
        with open(path) as f:
            records.append(json.load(f))
    return records


def make_timing_summary_from_replica_json(
    result_glob: str | Path,
    wall_clock_total_sec: float,
    label: str,
    max_workers: int | None = None,
) -> dict:
    """Build a timing summary from PCS/EC2 replica JSON files."""
    aggregate_t0 = time.perf_counter()
    records = load_replica_records(result_glob)
    summary = summarize_timing(
        records=records,
        wall_clock_total_sec=wall_clock_total_sec,
        label=label,
        max_workers=max_workers,
    )
    summary["aggregate_elapsed_sec"] = time.perf_counter() - aggregate_t0
    summary["result_glob"] = str(result_glob)
    return summary


def compare_timing_summaries(ec2_summary: dict, pcs_summary: dict) -> dict:
    """Return comparison rows and EC2/PCS wall-clock speedup."""
    rows = []
    for summary in [ec2_summary, pcs_summary]:
        wall = float(summary.get("wall_clock_total_sec", 0.0))
        replica_sum = float(summary.get("sum_replica_elapsed_sec", 0.0))
        rows.append({
            "label": summary.get("label", "run"),
            "tasks": summary.get("total_tasks", ""),
            "workers": summary.get("max_workers", ""),
            "wall_clock_sec": wall,
            "sum_replica_sec": replica_sum,
            "parallelism_proxy": replica_sum / wall if wall else 0.0,
            "aggregate_sec": float(summary.get("aggregate_elapsed_sec", 0.0)),
        })

    speedup = (
        rows[0]["wall_clock_sec"] / rows[1]["wall_clock_sec"]
        if rows[0]["wall_clock_sec"] and rows[1]["wall_clock_sec"]
        else 0.0
    )
    return {"rows": rows, "wall_clock_speedup": speedup}


def format_timing_comparison(comparison: dict) -> str:
    """Format timing comparison rows as a plain-text table."""
    lines = [
        f"{'run':<18} {'tasks':>7} {'workers':>8} {'wall(s)':>12} "
        f"{'sum_replica(s)':>16} {'parallelism':>13} {'aggregate(s)':>13}",
        "-" * 92,
    ]
    for row in comparison["rows"]:
        lines.append(
            f"{row['label']:<18} {row['tasks']:>7} {str(row['workers']):>8} "
            f"{row['wall_clock_sec']:>12.2f} {row['sum_replica_sec']:>16.2f} "
            f"{row['parallelism_proxy']:>13.2f} {row['aggregate_sec']:>13.2f}"
        )
    lines.append("")
    lines.append(f"wall-clock speedup, EC2 / PCS: {comparison['wall_clock_speedup']:.2f}x")
    return "\n".join(lines)


def print_timing_comparison(ec2_summary: dict, pcs_summary: dict) -> None:
    """Print an EC2 versus PCS timing comparison table."""
    print(format_timing_comparison(compare_timing_summaries(ec2_summary, pcs_summary)))
