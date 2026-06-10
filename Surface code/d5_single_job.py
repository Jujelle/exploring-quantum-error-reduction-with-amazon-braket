#!/usr/bin/env python3
"""Single d=5 surface-code test job for AWS PCS.

This is intentionally small and direct: one Slurm job, one d=5 point, one
output JSON file. Use it to verify the PCS Python environment and
Stim/PyMatching pipeline before submitting job arrays.
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


def run(distance: int, rounds: int, p: float, shots: int, seed: int) -> dict:
    circuit = build_circuit(distance=distance, rounds=rounds, p=p)
    data_qubits, ancilla_qubits, active_qubits = count_active_qubits(circuit)

    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler(seed=seed)

    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    pred = matcher.decode_batch(dets)
    if pred.ndim == 1:
        pred = pred.reshape((shots, 1))

    logical_errors = int(np.sum(np.any(pred != obs, axis=1)))
    return {
        "distance": distance,
        "rounds": rounds,
        "physical_error_rate": p,
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
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", default="results/d5_single_job.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    record = run(
        distance=5,
        rounds=args.rounds,
        p=args.p,
        shots=args.shots,
        seed=args.seed,
    )
    record["elapsed_sec"] = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(record, f, indent=2)

    print("[done] d=5 single-job test")
    print(f"  p                  = {record['physical_error_rate']}")
    print(f"  rounds             = {record['rounds']}")
    print(f"  shots              = {record['shots']}")
    print(f"  active qubits      = {record['active_qubits']}")
    print(f"  detectors          = {record['detectors']}")
    print(f"  matching edges     = {record['matching_edges']}")
    print(f"  logical errors     = {record['logical_errors']}")
    print(f"  logical error rate = {record['logical_error_rate']:.6%}")
    print(f"  elapsed sec        = {record['elapsed_sec']:.2f}")
    print(f"  output             = {args.output}")


if __name__ == "__main__":
    main()

