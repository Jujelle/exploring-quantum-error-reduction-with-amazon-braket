from collections import Counter
import matplotlib.pyplot as plt
from braket.aws import AwsDevice
from braket.circuits import Circuit
from braket.tracking import Tracker

# =================================================================
# Full Shor 9-qubit QEC (Quantum Error Correction) demo
#   Data qubits      : qubits 0..8
#   Z-stabilizers    : ancillas 9..14   (detects X errors)
#   X-stabilizers    : ancillas 15, 16  (detects Z errors)
# =================================================================
P_X, P_Z = 0.02, 0.02
SHOTS    = 4000

# ---- Circuit Fragments ------------------------------------------
def add_encoding(qc):                      # Pure encoding; q0 starts as the logical input state
    qc.cnot(0, 3); qc.cnot(0, 6)
    qc.h(0); qc.h(3); qc.h(6)
    qc.cnot(0, 1); qc.cnot(0, 2)
    qc.cnot(3, 4); qc.cnot(3, 5)
    qc.cnot(6, 7); qc.cnot(6, 8)

def add_decoding(qc):                      # Inverse of encoding
    qc.cnot(6, 8); qc.cnot(6, 7)
    qc.cnot(3, 5); qc.cnot(3, 4)
    qc.cnot(0, 2); qc.cnot(0, 1)
    qc.h(6); qc.h(3); qc.h(0)
    qc.cnot(0, 6); qc.cnot(0, 3)

def add_z_syndromes(qc):                   # 6 ZZ stabilizers (a9..a14)
    a = list(range(9, 15)); qc.h(a)
    qc.cz(9, 0);  qc.cz(9, 1)
    qc.cz(10, 1); qc.cz(10, 2)
    qc.cz(11, 3); qc.cz(11, 4)
    qc.cz(12, 4); qc.cz(12, 5)
    qc.cz(13, 6); qc.cz(13, 7)
    qc.cz(14, 7); qc.cz(14, 8)
    qc.h(a)

def add_x_syndromes(qc):                   # 2 X⊗⁶ stabilizers (a15, a16)
    qc.h([15, 16])
    qc.cnot(15, 0); qc.cnot(15, 1); qc.cnot(15, 2)
    qc.cnot(15, 3); qc.cnot(15, 4); qc.cnot(15, 5)
    qc.cnot(16, 3); qc.cnot(16, 4); qc.cnot(16, 5)
    qc.cnot(16, 6); qc.cnot(16, 7); qc.cnot(16, 8)
    qc.h([15, 16])

def add_noise(qc):
    for q in range(9):
        qc.bit_flip(q, P_X)                 # Inject both error types simultaneously
        qc.phase_flip(q, P_Z)

# ---- Circuit Assembly -------------------------------------------
def build_circuit(basis):
    qc = Circuit()
    if basis == 'X': qc.h(0)                # |0> → |+> for X-basis test
    add_encoding(qc)
    add_noise(qc)
    add_z_syndromes(qc)
    add_x_syndromes(qc)
    add_decoding(qc)
    if basis == 'X': qc.h(0)                # Rotate back to Z-basis before measurement
    return qc

# ---- Classical Syndrome → Correction Logic -----------------------
# Z-syndrome (a9..a14) mapping → identify which data qubit had an X error
X_QUBIT = {
    (1,0,0,0,0,0): 0, (1,1,0,0,0,0): 1, (0,1,0,0,0,0): 2,
    (0,0,1,0,0,0): 3, (0,0,1,1,0,0): 4, (0,0,0,1,0,0): 5,
    (0,0,0,0,1,0): 6, (0,0,0,0,1,1): 7, (0,0,0,0,0,1): 8,
}
# X-syndrome (a15, a16) mapping → identify which block had a Z error
Z_BLOCK = {(1,0): 0, (1,1): 1, (0,1): 2}

def classical_correct(bs, basis):
    """Determines whether to flip the q0 readout based on measured syndromes."""
    q0     = int(bs[0])
    syn_z  = tuple(int(b) for b in bs[9:15])
    syn_x  = tuple(int(b) for b in bs[15:17])
    flip   = 0
    if basis == 'X':
        # Logical X readout ⇔ anticommutes with X_L = Z0Z3Z6 
        # ⇔ An X error occurred on q ∈ {0,3,6}
        k = X_QUBIT.get(syn_z)
        if k in (0, 3, 6): flip ^= 1
    else:  # 'Z' basis
        # Logical Z readout ⇔ anticommutes with Z_L = X0X1X2 
        # ⇔ A Z error occurred in block 0
        if Z_BLOCK.get(syn_x) == 0: flip ^= 1
    return q0 ^ flip

# ---- Execution --------------------------------------------------
device  = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/dm1")
tracker = Tracker().start()

results = {}
for basis, expected in [('X', 0), ('Z', 0)]:
    label = '|+>_L (Testing X-error correction)' if basis == 'X' else '|0>_L (Testing Z-error correction)'
    print(f"\n=== Test {basis}: {label} ===")
    task = device.run(build_circuit(basis), shots=SHOTS)
    print(f"Task ARN : {task.id}")
    res = task.result()
    print(f"Status   : {task.state()}")

    n_raw = n_corr = 0
    for bs, c in res.measurement_counts.items():
        if int(bs[0])                    == expected: n_raw  += c
        if classical_correct(bs, basis)  == expected: n_corr += c
    results[basis] = (n_raw / SHOTS, n_corr / SHOTS)
    print(f"  Raw  fidelity : {n_raw / SHOTS:.4f}")
    print(f"  Corr fidelity : {n_corr / SHOTS:.4f}")

print(f"\nEstimated cost: ${tracker.simulator_tasks_cost():.4f} USD")

# ---- Visualization ----------------------------------------------
labels = ['|+>_L\n(X-error correction)', '|0>_L\n(Z-error correction)']
raws   = [results['X'][0], results['Z'][0]]
corrs  = [results['X'][1], results['Z'][1]]

x = list(range(2))
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar([i - 0.2 for i in x], raws,  width=0.4, label='No correction', color='salmon')
ax.bar([i + 0.2 for i in x], corrs, width=0.4, label='Corrected',     color='skyblue')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.05); ax.set_ylabel('Logical fidelity')
ax.set_title(f'Shor 9-qubit code (P_X = P_Z = {P_X})')
ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.savefig('test.png')