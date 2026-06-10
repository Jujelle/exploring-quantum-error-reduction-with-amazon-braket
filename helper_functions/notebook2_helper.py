"""Helper functions for Notebook 2: T1/T2 characterization and DD.

This module keeps circuit builders, fit models, and reusable plotting/job helpers
out of the notebook so the notebook can focus on parameters and results.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from braket.circuits import Circuit
from braket.devices import LocalSimulator


def build_noisy_t1_circuit(tau, gamma):
    """Build an excited qubit followed by amplitude-damped idle steps."""
    t1_circuit = Circuit()
    t1_circuit.x(0)

    for _ in range(tau):
        t1_circuit.i(0)
        t1_circuit.amplitude_damping(target=0, gamma=gamma)

    return t1_circuit


def exp_decay(t, A, T, C):
    """Single exponential decay with offset."""
    return A * np.exp(-t / T) + C


def build_noisy_ramsey_circuit(tau, gamma, gamma_ph, delta):
    """Build a noisy Ramsey circuit with quasi-static detuning."""
    ramsey_circuit = Circuit()
    ramsey_circuit.h(0)

    for _ in range(tau):
        ramsey_circuit.i(0)
        ramsey_circuit.rz(0, delta)
        ramsey_circuit.amplitude_damping(target=0, gamma=gamma)
        ramsey_circuit.phase_damping(target=0, gamma=gamma_ph)

    ramsey_circuit.h(0)
    return ramsey_circuit


def ramsey_decay_with_quasistatic_noise(t, A, T2_markovian, T2_star):
    """Ramsey contrast model with exponential and Gaussian decay envelopes."""
    return 0.5 * (
        1
        + A
        * np.exp(-t / T2_markovian)
        * np.exp(-(t / T2_star) ** 2)
    )


def add_noisy_free_precession(circuit, steps, gamma, gamma_ph, delta):
    """Append noisy idle/precession steps to an existing circuit."""
    for _ in range(steps):
        circuit.i(0)
        circuit.rz(0, delta)
        circuit.amplitude_damping(target=0, gamma=gamma)
        circuit.phase_damping(target=0, gamma=gamma_ph)


def build_noisy_hahn_echo_circuit(tau, gamma, gamma_ph, delta):
    """Build H - tau/2 - X - tau/2 - H with noisy free precession."""
    hahn_echo_circuit = Circuit()
    tau_half = tau // 2

    hahn_echo_circuit.h(0)
    add_noisy_free_precession(hahn_echo_circuit, tau_half, gamma, gamma_ph, delta)
    hahn_echo_circuit.x(0)
    add_noisy_free_precession(
        hahn_echo_circuit,
        tau - tau_half,
        gamma,
        gamma_ph,
        delta,
    )
    hahn_echo_circuit.h(0)

    return hahn_echo_circuit


def build_noisy_cpmg_circuit(tau, gamma, gamma_ph, delta, n_pulses=4):
    """Build a noisy CPMG sequence with evenly distributed refocusing pulses."""
    cpmg_circuit = Circuit()
    cpmg_circuit.h(0)

    half_blocks = [tau // (2 * n_pulses)] * (2 * n_pulses)
    for block_index in range(tau - sum(half_blocks)):
        half_blocks[block_index % len(half_blocks)] += 1

    for pulse_index in range(n_pulses):
        add_noisy_free_precession(
            cpmg_circuit,
            half_blocks[2 * pulse_index],
            gamma,
            gamma_ph,
            delta,
        )
        cpmg_circuit.x(0)
        add_noisy_free_precession(
            cpmg_circuit,
            half_blocks[2 * pulse_index + 1],
            gamma,
            gamma_ph,
            delta,
        )

    cpmg_circuit.h(0)
    return cpmg_circuit


def average_p0_over_detunings(build_circuit, taus, detuning_samples, device=None):
    """Average P(|0>) over sampled detunings for each tau."""
    if device is None:
        device = LocalSimulator("braket_dm")

    p0_values = []

    for tau_index, tau in enumerate(taus):
        if int(tau) == 0:
            p0_values.append(1.0)
            continue

        p0_for_tau = []

        for delta in detuning_samples[tau_index]:
            qc = build_circuit(int(tau), float(delta)).probability(target=[0])
            result = device.run(qc, shots=0).result()
            p0_for_tau.append(result.values[0][0])

        p0_values.append(np.mean(p0_for_tau))

    return np.array(p0_values)


def single_exp(t, p_inf, A, T2):
    """Single exponential approach to an asymptote."""
    return p_inf + A * np.exp(-t / T2)


def single_exp(t, p_inf, A, T2):
    return p_inf + A * np.exp(-t / T2)

def ramsey_exp_cos(t, p_inf, A, T2star, f_det, phi):
    return p_inf + A * np.exp(-t / T2star) * np.cos(2 * np.pi * f_det * t + phi)


def plot_hybrid_t2_fits(evolution_times_us_hybrid, p0_qpu_hybrid):
    """Fit Ramsey with detuned oscillatory decay; fit Hahn echo and CPMG4 with single exponential."""

    colors = {
        "Ramsey": "#0072B2",
        "Hahn echo": "#D55E00",
        "CPMG4": "#009E73",
    }
    markers = {
        "Ramsey": "o",
        "Hahn echo": "s",
        "CPMG4": "^",
    }
    linestyles = {
        "Ramsey": "-",
        "Hahn echo": "--",
        "CPMG4": "-.",
    }

    fit_results = {}
    plt.figure(figsize=(7, 4.5))

    for sequence_name in ["Ramsey", "Hahn echo", "CPMG4"]:
        t = np.array(evolution_times_us_hybrid, dtype=float)
        y = np.array(p0_qpu_hybrid[sequence_name], dtype=float)

        p_inf_guess = np.mean(y[-3:])
        A_guess = 0.5 * (np.max(y) - np.min(y))
        T_guess = (t[-1] - t[0]) / 2

        if sequence_name == "Ramsey":
            dt = np.mean(np.diff(t))
            y_fft = y - np.mean(y)
            freqs = np.fft.rfftfreq(len(t), d=dt)
            fft_amp = np.abs(np.fft.rfft(y_fft))
            f_guess = freqs[1 + np.argmax(fft_amp[1:])] if len(freqs) > 1 else 0.1
            phi_guess = 0.0

            popt, pcov = curve_fit(
                ramsey_exp_cos,
                t,
                y,
                p0=[p_inf_guess, A_guess, T_guess, f_guess, phi_guess],
                bounds=(
                    [0.0, -1.0, 0.0, 0.0, -2 * np.pi],
                    [1.0, 1.0, np.inf, np.inf, 2 * np.pi],
                ),
                maxfev=20000,
            )

            p_inf, A, T2star, f_det, phi = popt
            perr = np.sqrt(np.diag(pcov))

            fit_results[sequence_name] = {
                "p_inf": p_inf,
                "A": A,
                "T2star_us": T2star,
                "T2star_err_us": perr[2],
                "f_det_MHz": f_det,
                "f_det_err_MHz": perr[3],
                "phi_rad": phi,
            }

            t_fit = np.linspace(t.min(), t.max(), 500)
            y_fit = ramsey_exp_cos(t_fit, *popt)

            fit_label = rf"Ramsey fit, $T_2^*={T2star:.2f}\,\mu$s, $f={f_det:.2f}$ MHz"

        else:
            popt, pcov = curve_fit(
                single_exp,
                t,
                y,
                p0=[p_inf_guess, y[0] - y[-1], T_guess],
                bounds=([0.0, -1.0, 0.0], [1.0, 1.0, np.inf]),
                maxfev=10000,
            )

            p_inf, A, T2 = popt
            perr = np.sqrt(np.diag(pcov))

            fit_results[sequence_name] = {
                "p_inf": p_inf,
                "A": A,
                "T2_us": T2,
                "T2_err_us": perr[2],
            }

            t_fit = np.linspace(t.min(), t.max(), 500)
            y_fit = single_exp(t_fit, *popt)

            fit_label = rf"{sequence_name} fit, $T_2={T2:.2f}\,\mu$s"

        plt.plot(
            t,
            y,
            marker=markers[sequence_name],
            linestyle="none",
            color=colors[sequence_name],
            markersize=6,
            label=f"{sequence_name} data",
        )

        plt.plot(
            t_fit,
            y_fit,
            linestyle=linestyles[sequence_name],
            color=colors[sequence_name],
            linewidth=2,
            label=fit_label,
        )

    plt.xlabel(r"Evolution time $\tau$ ($\mu$s)", fontsize=14)
    plt.ylabel(r"$P(|0\rangle)$", fontsize=14)
    plt.title("Dynamical Decoupling T2 Fits", fontsize=15)
    plt.legend(fontsize=9, frameon=False)
    plt.tick_params(direction="in", labelsize=12)
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    return fit_results


def make_dd_real_qpu_hybrid_job(qpu_arn, job_role_arn):
    """Create the decorated Hybrid Job function after notebook config is set."""
    from math import pi

    from braket.aws import AwsDevice
    from braket.jobs import get_job_device_arn, hybrid_job
    from braket.jobs.metrics import log_metric
    from braket.pulse import PulseSequence

    @hybrid_job(device=qpu_arn, role_arn=job_role_arn, data_format="pickle")
    def dd_real_qpu_hybrid_job(
        qubit=1,
        shots=1000,
        evolution_times_us_csv="2,4,6,8,10,12",
        num_cpmg_pulses=4,
        data_bucket_name="amazon-braket-batch-project-338348453266-us-east-1",
        s3_prefix="qubit_t1_t2_characterization/dd_real_qpu_hybrid_job",
    ):
        """Run the direct-QPU dynamical decoupling sweep inside a Hybrid Job."""
        time_grid = 4e-9
        qpu = AwsDevice(get_job_device_arn())
        s3_folder = (data_bucket_name, s3_prefix)
        evolution_times_us = [
            float(value)
            for value in evolution_times_us_csv.split(",")
            if value
        ]

        def aligned_delay_blocks(total_time_s, n_blocks):
            """Split a total delay into hardware-aligned blocks."""
            total_steps = int(round(total_time_s / time_grid))
            if total_steps < n_blocks:
                raise ValueError("Total free-evolution time is too short.")

            base_steps, extra_steps = divmod(total_steps, n_blocks)
            delay_blocks = []
            for block_index in range(n_blocks):
                block_steps = base_steps
                if block_index < extra_steps:
                    block_steps += 1
                delay_blocks.append(block_steps * time_grid)
            return delay_blocks

        def idle_gate(qubit, idle_time_s):
            """Return a pulse gate that idles the selected qubit drive frame."""
            delay_steps = int(round(idle_time_s / time_grid))
            idle_time_s = delay_steps * time_grid

            drive_frame = qpu.frames[f"Transmon_{qubit}_charge_tx"]
            pulse_sequence = PulseSequence().delay(drive_frame, idle_time_s)

            return Circuit().pulse_gate(
                targets=[qubit],
                pulse_sequence=pulse_sequence,
                display_name="IDLE",
            )

        def add_h(circuit, qubit):
            """Append a native-gate decomposition of H to the circuit."""
            circuit.rz(qubit, pi / 2)
            circuit.rx(qubit, pi / 2)
            circuit.rz(qubit, pi / 2)

        def ramsey(qubit, total_free_time_s):
            """Build H - tau - H for Ramsey measurement."""
            (free_time_s,) = aligned_delay_blocks(total_free_time_s, 1)
            circuit = Circuit()
            add_h(circuit, qubit)
            circuit.add_circuit(idle_gate(qubit, free_time_s))
            add_h(circuit, qubit)
            return circuit

        def hahn_echo(qubit, total_free_time_s):
            """Build H - tau/2 - X - tau/2 - H for Hahn echo."""
            first_half_s, second_half_s = aligned_delay_blocks(total_free_time_s, 2)
            circuit = Circuit()
            add_h(circuit, qubit)
            circuit.add_circuit(idle_gate(qubit, first_half_s))
            circuit.rx(qubit, pi)
            circuit.add_circuit(idle_gate(qubit, second_half_s))
            add_h(circuit, qubit)
            return circuit

        def cpmg(qubit, total_free_time_s):
            """Build CPMG with evenly spaced X refocusing pulses."""
            delay_blocks_s = aligned_delay_blocks(
                total_free_time_s,
                2 * num_cpmg_pulses,
            )
            circuit = Circuit()
            add_h(circuit, qubit)

            for pulse_index in range(num_cpmg_pulses):
                circuit.add_circuit(idle_gate(qubit, delay_blocks_s[2 * pulse_index]))
                circuit.rx(qubit, pi)
                circuit.add_circuit(
                    idle_gate(qubit, delay_blocks_s[2 * pulse_index + 1])
                )

            add_h(circuit, qubit)
            return circuit

        sequences = [
            ("Ramsey", ramsey),
            ("Hahn echo", hahn_echo),
            ("CPMG4", cpmg),
        ]

        submitted_tasks = []

        for evolution_time_us in evolution_times_us:
            evolution_time_s = evolution_time_us * 1e-6

            for sequence_name, build_sequence in sequences:
                circuit = build_sequence(qubit, evolution_time_s)
                task = qpu.run(
                    circuit,
                    s3_destination_folder=s3_folder,
                    shots=shots,
                    poll_interval_seconds=5,
                )

                submitted_tasks.append(
                    {
                        "sequence": sequence_name,
                        "time_us": evolution_time_us,
                        "task": task,
                        "task_id": task.id,
                    }
                )

                print(
                    f"Submitted {sequence_name:9s} at tau = {evolution_time_us:>4g} us "
                    f"as task {task.id}"
                )

        p0_qpu = {sequence_name: [] for sequence_name, _ in sequences}
        task_ids_by_sequence = {sequence_name: [] for sequence_name, _ in sequences}

        for task_index, item in enumerate(submitted_tasks):
            result = item["task"].result()
            p0 = result.measurement_counts.get("0", 0) / shots

            p0_qpu[item["sequence"]].append(float(p0))
            task_ids_by_sequence[item["sequence"]].append(item["task_id"])

            log_metric(
                metric_name="p0_" + item["sequence"].lower().replace(" ", "_"),
                iteration_number=task_index,
                value=float(p0),
            )

            print(
                f"{item['sequence']:9s} tau = {item['time_us']:>4g} us: "
                f"P(|0>) = {p0:.3f}"
            )

        return {
            "device_arn": qpu.arn,
            "qubit": qubit,
            "shots": shots,
            "evolution_times_us": evolution_times_us,
            "p0_qpu": p0_qpu,
            "task_ids_by_sequence": task_ids_by_sequence,
            "submitted_tasks": [
                {
                    "sequence": item["sequence"],
                    "time_us": item["time_us"],
                    "task_id": item["task_id"],
                }
                for item in submitted_tasks
            ],
        }

    return dd_real_qpu_hybrid_job


__all__ = [
    "add_noisy_free_precession",
    "average_p0_over_detunings",
    "build_noisy_cpmg_circuit",
    "build_noisy_hahn_echo_circuit",
    "build_noisy_ramsey_circuit",
    "build_noisy_t1_circuit",
    "exp_decay",
    "make_dd_real_qpu_hybrid_job",
    "plot_hybrid_t2_fits",
    "ramsey_decay_with_quasistatic_noise",
    "single_exp",
]
