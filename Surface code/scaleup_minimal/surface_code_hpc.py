#!/usr/bin/env python3
"""Run one surface-code Monte Carlo replica and write one JSON result.

Designed for AWS PCS / Slurm job arrays. Each task receives one distance,
one physical error rate, one seed, and one shot count.
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import pymatching
import stim


def build_circuit(distance: int, rounds: int, p: float) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def count_active_qubits(circuit: stim.Circuit) -> tuple[int, int, int]:
    data = 0
    ancilla = 0
    for xy in circuit.get_final_qubit_coordinates().values():
        x, y = int(xy[0]), int(xy[1])
        if x % 2 == 1 and y % 2 == 1:
            data += 1
        else:
            ancilla += 1
    return data, ancilla, data + ancilla


def run_replica(distance: int, rounds: int, p: float, shots: int, seed: int) -> dict:
    circuit = build_circuit(distance=distance, rounds=rounds, p=p)
    data_qubits, ancilla_qubits, active_qubits = count_active_qubits(circuit)

    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots=shots, separate_observables=True)
    pred = matcher.decode_batch(det)
    if pred.ndim == 1:
        pred = pred.reshape((shots, 1))

    logical_errors = int(np.sum(np.any(pred != obs, axis=1)))
    return {
        "distance": distance,
        "rounds": rounds,
        "physical_error_rate": p,
        "error_rate": p,
        "shots": shots,
        "seed": seed,
        "data_qubits": data_qubits,
        "ancilla_qubits": ancilla_qubits,
        "active_qubits": active_qubits,
        "stim_label_span": circuit.num_qubits,
        "detectors": circuit.num_detectors,
        "matching_edges": matcher.num_edges,
        "logical_errors": logical_errors,
        "logical_error_rate": logical_errors / shots if shots else 0.0,
        "stim_version": stim.__version__,
        "pymatching_version": pymatching.__version__,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--error-rate", type=float, required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replica", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rounds = args.rounds if args.rounds is not None else args.distance

    t0 = time.time()
    record = run_replica(
        distance=args.distance,
        rounds=rounds,
        p=args.error_rate,
        shots=args.shots,
        seed=args.seed,
    )
    record.update({
        "replica": args.replica,
        "elapsed_sec": time.time() - t0,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        "slurm_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(record, f, indent=2)

    print(
        f"[done] d={record['distance']} p={record['physical_error_rate']:.6f} "
        f"replica={record['replica']} shots={record['shots']} "
        f"errors={record['logical_errors']} "
        f"L_err={record['logical_error_rate']:.6%} "
        f"elapsed={record['elapsed_sec']:.2f}s"
    )
    print(f"[json] {args.output}")


if __name__ == "__main__":
    main()
