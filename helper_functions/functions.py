from braket.circuits import Circuit
import numpy as np


def rzz_angle(i,j, theta):
    """ implement e^(i  theta Z_i Z_j) """
    return Circuit().cnot(i,j).rz(j, -2 *theta).cnot(i,j)

def ising_1d(self_interaction : float, hopping : float, num_qubits : int) -> Circuit:
    circ = Circuit()
    for i in range(num_qubits):
        circ.rx(i,-2*self_interaction)
    for i in range(0,num_qubits-1,2):
        circ+= rzz_angle(i,i+1, hopping)
    for i in range(1,num_qubits-1,2):
        circ+= rzz_angle(i,i+1, hopping)
    return circ

def fit_exp_linearized(x, y, a, eps=1e-12):

    # determine sign automatically
    sign = np.sign(np.mean(y - a))

    # shifted positive data
    z = sign * (y - a)

    # remove invalid points
    mask = z > eps

    xfit = x[mask]
    zfit = np.log(z[mask])

    # linear regression
    slope, intercept = np.polyfit(xfit, zfit, 1)

    c = -slope
    b = sign * np.exp(intercept)

    return [b,c]