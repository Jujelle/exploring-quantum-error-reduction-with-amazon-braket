#!/usr/bin/env python3
"""Complete executable Shor-code DM1 demo.

Example:
    python3 run_shor_dm1_demo.py --mode x-only --logical-input 1 --shots 1000 --p-error 0.03
    python3 run_shor_dm1_demo.py --mode full --shots 4000 --p-x 0.02 --p-z 0.02
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from braket.aws import AwsDevice
from braket.tracking import Tracker

from shor_code_helpers import (
    build_full_shor_qec_circuit,
    build_shor_x_error_demo_circuit,
    decode_x_error_counts,
    format_task_report,
    plot_full_qec_summary,
    plot_raw_corrected_counts,
    score_full_qec_counts,
)


DM1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/dm1"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Shor-code demos on Amazon Braket DM1.")
    parser.add_argument("--mode", choices=["x-only", "full"], default="x-only")
    parser.add_argument("--logical-input", choices=["0", "1"], default="1")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--p-error", type=float, default=0.03, help="X-error probability for x-only mode.")
    parser.add_argument("--p-x", type=float, default=0.02, help="Bit-flip probability for full mode.")
    parser.add_argument("--p-z", type=float, default=0.02, help="Phase-flip probability for full mode.")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def run_x_only(device, tracker, args):
    circuit = build_shor_x_error_demo_circuit(
        logical_input=args.logical_input,
        p_error=args.p_error,
    )
    print("Submitting X-error-only Shor demo to DM1 ...")
    task = device.run(circuit, shots=args.shots)
    print(f"Task ARN: {task.id}")
    result = task.result()
    print(format_task_report(result, tracker=tracker))

    decoded = decode_x_error_counts(result.measurement_counts, logical_input=args.logical_input)
    print(f"Expected data   : {decoded['expected_data']}")
    print(f"Raw recovery    : {decoded['raw_success']}/{decoded['total']} = {decoded['raw_rate']:.4f}")
    print(
        f"Corrected       : {decoded['corrected_success']}/{decoded['total']} "
        f"= {decoded['corrected_rate']:.4f}"
    )
    if not args.no_plot:
        plot_raw_corrected_counts(decoded["raw_counts"], decoded["corrected_counts"])
        plt.show()


def run_full(device, tracker, args):
    results = {}
    for basis, expected in [("X", 0), ("Z", 0)]:
        label = "|+>_L" if basis == "X" else "|0>_L"
        print(f"Submitting full Shor demo for {label} to DM1 ...")
        task = device.run(build_full_shor_qec_circuit(basis, p_x=args.p_x, p_z=args.p_z), shots=args.shots)
        print(f"Task ARN: {task.id}")
        result = task.result()
        print(format_task_report(result, tracker=tracker))

        score = score_full_qec_counts(result.measurement_counts, basis=basis, expected=expected)
        results[basis] = (score["raw_fidelity"], score["corrected_fidelity"])
        print(f"Raw fidelity       : {score['raw_fidelity']:.4f}")
        print(f"Corrected fidelity : {score['corrected_fidelity']:.4f}")

    if not args.no_plot:
        plot_full_qec_summary(results, p_x=args.p_x, p_z=args.p_z)
        plt.show()


def main():
    args = parse_args()
    tracker = Tracker().start()
    device = AwsDevice(DM1_ARN)

    if args.mode == "x-only":
        run_x_only(device, tracker, args)
    else:
        run_full(device, tracker, args)


if __name__ == "__main__":
    main()
