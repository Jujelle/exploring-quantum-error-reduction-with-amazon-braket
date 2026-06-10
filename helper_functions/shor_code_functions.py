#!/usr/bin/env python3
"""Reusable Shor-code helpers for Notebook 4 and executable demos."""
from __future__ import annotations

import datetime as dt
from collections import Counter
from typing import Mapping

import matplotlib.pyplot as plt
from braket.circuits import Circuit, Noise
from braket.circuits.noise_model import GateCriteria, NoiseModel


X_ERROR_MASK_BY_Z_SYNDROME = {
    (0, 0, 0, 0, 0, 0): "000000000",
    (1, 0, 0, 0, 0, 0): "011000000",
    (1, 1, 0, 0, 0, 0): "010000000",
    (0, 1, 0, 0, 0, 0): "001000000",
    (0, 0, 1, 0, 0, 0): "000011000",
    (0, 0, 1, 1, 0, 0): "000010000",
    (0, 0, 0, 1, 0, 0): "000001000",
    (0, 0, 0, 0, 1, 0): "000000011",
    (0, 0, 0, 0, 1, 1): "000000010",
    (0, 0, 0, 0, 0, 1): "000000001",
}

X_QUBIT_BY_Z_SYNDROME = {
    (1, 0, 0, 0, 0, 0): 0,
    (1, 1, 0, 0, 0, 0): 1,
    (0, 1, 0, 0, 0, 0): 2,
    (0, 0, 1, 0, 0, 0): 3,
    (0, 0, 1, 1, 0, 0): 4,
    (0, 0, 0, 1, 0, 0): 5,
    (0, 0, 0, 0, 1, 0): 6,
    (0, 0, 0, 0, 1, 1): 7,
    (0, 0, 0, 0, 0, 1): 8,
}

Z_BLOCK_BY_X_SYNDROME = {
    (1, 0): 0,
    (1, 1): 1,
    (0, 1): 2,
}


def normalize_logical_input(logical_input: str | int) -> str:
    """Accept 0/1 as strings or ints, and return '0' or '1'."""
    logical_input = str(logical_input)
    if logical_input not in {"0", "1"}:
        raise ValueError("logical_input must be '0' or '1'")
    return logical_input


def expected_decoded_data(logical_input: str | int) -> str:
    """Expected data bits after ideal Shor decode."""
    return "100000000" if normalize_logical_input(logical_input) == "1" else "000000000"


def prepare_logical_input(qc: Circuit, logical_input: str | int) -> Circuit:
    """Prepare q0 as |0> or |1> before Shor encoding."""
    if normalize_logical_input(logical_input) == "1":
        qc.x(0)
    return qc


def add_shor_encoding(qc: Circuit) -> Circuit:
    """Add Shor 9-qubit encoding to a circuit."""
    qc.cnot(0, 3).cnot(0, 6)
    qc.h([0, 3, 6])
    qc.cnot(0, 1).cnot(0, 2)
    qc.cnot(3, 4).cnot(3, 5)
    qc.cnot(6, 7).cnot(6, 8)
    return qc


def add_shor_decoding(qc: Circuit) -> Circuit:
    """Add inverse Shor encoding for decoded data-bit readout demos."""
    qc.cnot(6, 8).cnot(6, 7)
    qc.cnot(3, 5).cnot(3, 4)
    qc.cnot(0, 2).cnot(0, 1)
    qc.h([6, 3, 0])
    qc.cnot(0, 6).cnot(0, 3)
    return qc


def add_z_syndromes(qc: Circuit) -> Circuit:
    """Measure the six Z-type stabilizers using ancillas 9..14."""
    ancillas = list(range(9, 15))
    qc.h(ancillas)
    qc.cz(9, 0).cz(9, 1)
    qc.cz(10, 1).cz(10, 2)
    qc.cz(11, 3).cz(11, 4)
    qc.cz(12, 4).cz(12, 5)
    qc.cz(13, 6).cz(13, 7)
    qc.cz(14, 7).cz(14, 8)
    qc.h(ancillas)
    return qc


def add_x_syndromes(qc: Circuit) -> Circuit:
    """Measure the two X-type stabilizers using ancillas 15 and 16."""
    qc.h([15, 16])
    qc.cnot(15, 0).cnot(15, 1).cnot(15, 2)
    qc.cnot(15, 3).cnot(15, 4).cnot(15, 5)
    qc.cnot(16, 3).cnot(16, 4).cnot(16, 5)
    qc.cnot(16, 6).cnot(16, 7).cnot(16, 8)
    qc.h([15, 16])
    return qc


def add_independent_xz_noise(qc: Circuit, p_x: float, p_z: float) -> Circuit:
    """Add independent bit-flip and phase-flip channels on data qubits."""
    for q in range(9):
        qc.bit_flip(q, p_x)
        qc.phase_flip(q, p_z)
    return qc


def build_shor_syndrome_circuit(logical_input: str | int = "1") -> Circuit:
    """Build a 17-qubit Shor circuit with stabilizer measurements."""
    qc = Circuit()
    prepare_logical_input(qc, logical_input)
    add_shor_encoding(qc)
    add_z_syndromes(qc)
    add_x_syndromes(qc)
    return qc


def build_shor_x_error_demo_circuit(logical_input: str | int = "1", p_error: float = 0.03) -> Circuit:
    """Build a compact Shor demo circuit focused on X-error correction."""
    qc = Circuit()
    prepare_logical_input(qc, logical_input)
    add_shor_encoding(qc)
    for q in range(9):
        qc.bit_flip(q, p_error)
    add_z_syndromes(qc)
    add_shor_decoding(qc)
    return qc


def build_full_shor_qec_circuit(basis: str, p_x: float = 0.02, p_z: float = 0.02) -> Circuit:
    """Build the two-basis full Shor QEC demo circuit."""
    if basis not in {"X", "Z"}:
        raise ValueError("basis must be 'X' or 'Z'")
    qc = Circuit()
    if basis == "X":
        qc.h(0)
    add_shor_encoding(qc)
    add_independent_xz_noise(qc, p_x=p_x, p_z=p_z)
    add_z_syndromes(qc)
    add_x_syndromes(qc)
    add_shor_decoding(qc)
    if basis == "X":
        qc.h(0)
    return qc


def build_shor_noise_model(p_error: float = 0.10) -> NoiseModel:
    """Build a global bit-flip plus phase-flip noise model for the 9 data qubits."""
    noise_model = NoiseModel()
    noise_model.add_noise(Noise.BitFlip(probability=p_error), GateCriteria(qubits=range(9)))
    noise_model.add_noise(Noise.PhaseFlip(probability=p_error), GateCriteria(qubits=range(9)))
    return noise_model


def decode_x_error_counts(
    measurement_counts: Mapping[str, int],
    logical_input: str | int = "1",
) -> dict:
    """Decode the X-error-only Shor demo from measurement counts."""
    expected = expected_decoded_data(logical_input)
    raw_counts = Counter()
    corrected_counts = Counter()
    raw_success = 0
    corrected_success = 0
    total = 0

    for bitstring, count in measurement_counts.items():
        data = bitstring[0:9]
        syn_z = tuple(int(b) for b in bitstring[9:15])
        mask = X_ERROR_MASK_BY_Z_SYNDROME.get(syn_z, "000000000")
        corrected = "".join(str(int(d) ^ int(m)) for d, m in zip(data, mask))

        raw_counts[data] += count
        corrected_counts[corrected] += count
        total += count
        raw_success += count if data == expected else 0
        corrected_success += count if corrected == expected else 0

    return {
        "expected_data": expected,
        "raw_counts": raw_counts,
        "corrected_counts": corrected_counts,
        "total": total,
        "raw_success": raw_success,
        "corrected_success": corrected_success,
        "raw_rate": raw_success / total if total else 0.0,
        "corrected_rate": corrected_success / total if total else 0.0,
    }


def decode_batch_counts(measurement_counts: Mapping[str, int]) -> tuple[Counter, Counter]:
    """Apply simple X-error correction to Shor measurement counts."""
    raw_counts = Counter()
    corrected_counts = Counter()

    for bitstring, count in measurement_counts.items():
        data_bits = [int(b) for b in bitstring[0:9]]
        syn_z = tuple(int(b) for b in bitstring[9:15])
        mask = X_ERROR_MASK_BY_Z_SYNDROME.get(syn_z, "000000000")
        corrected = [bit ^ int(mask[i]) for i, bit in enumerate(data_bits)]

        raw_counts["".join(str(b) for b in data_bits)] += count
        corrected_counts["".join(str(b) for b in corrected)] += count

    return raw_counts, corrected_counts


def classical_correct_q0(bitstring: str, basis: str) -> int:
    """Return corrected q0 readout for the two-basis full Shor demo."""
    q0 = int(bitstring[0])
    syn_z = tuple(int(b) for b in bitstring[9:15])
    syn_x = tuple(int(b) for b in bitstring[15:17])
    flip = 0

    if basis == "X":
        qubit = X_QUBIT_BY_Z_SYNDROME.get(syn_z)
        if qubit in (0, 3, 6):
            flip ^= 1
    elif basis == "Z":
        if Z_BLOCK_BY_X_SYNDROME.get(syn_x) == 0:
            flip ^= 1
    else:
        raise ValueError("basis must be 'X' or 'Z'")

    return q0 ^ flip


def score_full_qec_counts(measurement_counts: Mapping[str, int], basis: str, expected: int = 0) -> dict:
    """Score raw and corrected logical fidelity for one full Shor demo basis."""
    total = sum(measurement_counts.values())
    raw_success = 0
    corrected_success = 0
    for bitstring, count in measurement_counts.items():
        if int(bitstring[0]) == expected:
            raw_success += count
        if classical_correct_q0(bitstring, basis) == expected:
            corrected_success += count
    return {
        "basis": basis,
        "expected": expected,
        "total": total,
        "raw_success": raw_success,
        "corrected_success": corrected_success,
        "raw_fidelity": raw_success / total if total else 0.0,
        "corrected_fidelity": corrected_success / total if total else 0.0,
    }


def parse_braket_time(value):
    """Parse Braket metadata timestamps when they are available."""
    if not value:
        return None
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def braket_metadata_duration_sec(task_metadata) -> float | None:
    """Return createdAt-to-endedAt duration from Braket task metadata."""
    start_time = parse_braket_time(getattr(task_metadata, "createdAt", None))
    end_time = parse_braket_time(getattr(task_metadata, "endedAt", None))
    return (end_time - start_time).total_seconds() if start_time and end_time else None


def format_task_report(result, tracker=None, notebook_wall_sec: float | None = None) -> str:
    """Format a cost/time report for an Amazon Braket task result."""
    metadata = result.task_metadata
    lines = [
        "=== Resource Cost & Tracking Report ===",
        f"Total Shots          : {metadata.shots}",
    ]
    if notebook_wall_sec is not None:
        lines.append(f"Notebook Wall Time   : {notebook_wall_sec:.2f} seconds")

    duration_sec = braket_metadata_duration_sec(metadata)
    if duration_sec is None:
        lines.append("Task Metadata Time   : unavailable from task metadata")
    else:
        lines.append(f"Task Metadata Time   : {duration_sec:.2f} seconds")

    if tracker is not None:
        lines.append(f"Estimated Cost       : ${tracker.simulator_tasks_cost():.4f} USD")
    lines.append("=======================================")
    return "\n".join(lines)


def plot_raw_corrected_counts(raw_counts, corrected_counts, title_prefix: str = ""):
    """Plot raw versus corrected data-bit distributions."""
    top_raw = dict(Counter(raw_counts).most_common(10))
    top_corr = dict(Counter(corrected_counts).most_common(10))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(top_raw.keys(), top_raw.values(), color="salmon", edgecolor="black")
    ax1.set_title(f"{title_prefix}Raw data bits (top 10)".strip())
    ax2.bar(top_corr.keys(), top_corr.values(), color="skyblue", edgecolor="black")
    ax2.set_title(f"{title_prefix}Corrected data bits (top 10)".strip())
    for ax in (ax1, ax2):
        ax.set_ylabel("Counts")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    return fig


def plot_full_qec_summary(results: Mapping[str, tuple[float, float]], p_x: float, p_z: float):
    """Plot raw/corrected logical fidelities for X- and Z-basis Shor demos."""
    labels = ["$|+>_L$\n(X-error correction)", "$|0>_L$\n(Z-error correction)"]
    raws = [results["X"][0], results["Z"][0]]
    corrected = [results["X"][1], results["Z"][1]]

    x = list(range(2))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - 0.2 for i in x], raws, width=0.4, label="No correction", color="salmon")
    ax.bar([i + 0.2 for i in x], corrected, width=0.4, label="Corrected", color="skyblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Logical fidelity", fontsize=20)
    ax.set_title(f"Shor 9-qubit code ($P_X$ = {p_x}, $P_Z$ = {p_z})", fontsize=24, pad=15)
    ax.legend(loc="lower right", fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig
