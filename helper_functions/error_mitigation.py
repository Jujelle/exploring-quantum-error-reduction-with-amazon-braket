import json, os
import numpy as np

from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.emulation.local_emulator import LocalEmulator
from mitiq.zne import construct_circuits

import time

json_path = os.path.join(os.path.dirname(__file__), "emerald_properties_251106.json")
with open(json_path, "r") as fp:
    props = json.load(fp)

emulator = LocalEmulator.from_json(props)
qd = LocalSimulator("braket_dm",noise_model= emulator._noise_model)

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

# def run(self_interaction : float, hopping : float, num_shots : int , num_qubits: int, index_steps: int, scale_factors):
def run(params: dict):
    self_interaction = params["self_interaction"] # the Hamiltonian that acts on a single spin, independent of its neighbors [in the Ising model]
    hopping          = params["hopping"] # the hopping allow "particles" to physically hop from one qubit to its neighbors [in the Ising model]
    num_shots        = params["num_shots"] # number of measurements M
    num_qubits       = params["num_qubits"] # number of qubits 
    index_steps      = params["index_steps"] # how long do we want to evolve the system (each index step represents a discrete unit time)
    scale_factors    = params["scale_factors"] # how we scale up the noises. Scale_factors = 1 means no extra pairs of gate to scale errors.
    extrapolation_method = params["extrapolation_method"]
    
    start_time = time.perf_counter()
    zne_evs = np.zeros((len(scale_factors),num_qubits))
    extrapolate_value = []
 
    ## Build up the Ising model circuit
    time_step = ising_1d(self_interaction,hopping, num_qubits)
    circ = Circuit()
    
    for step in range(index_steps+1):
        circ+= time_step 
    ## Apply random extra pairs of gate to scale errors in the circuit.
    noisy_circ = construct_circuits(circ, scale_factors)
    
    print(f'Running step: {step}')
    for j,item in enumerate(noisy_circ):
        print(f'  - subcircuit {j}')
        res = qd.run(item, shots = num_shots).result()
        ## map from 0 and 1 state to +1 and -1 on the averaged outcome
        zne_evs[j,:] = np.mean(res.measurements*(-2)+1, axis=0)
        
    ## fitting the ZNE coefficeint  
    if extrapolation_method == 'Richardson':
        extrapolate_richard = np.zeros((len(scale_factors),num_qubits))
        for j in range(num_qubits):
            coeff = np.polyfit(scale_factors, zne_evs[:,j].tolist(), len(scale_factors)-1)
            extrapolate_richard[:,j] = coeff
        extrapolate_value = extrapolate_richard
        
    if extrapolation_method == 'Linear':
        extrapolate_linear = np.zeros((2,num_qubits))
        for j in range(num_qubits):
            coeff = np.polyfit(scale_factors, zne_evs[:,j].tolist(), 1)
            extrapolate_linear[:,j] = coeff
        extrapolate_value = extrapolate_linear
    if extrapolation_method == 'Exp':
        extrapolate_exp = np.zeros((2,num_qubits))
        for j in range(num_qubits):
            coeff = fit_exp_linearized(np.array(scale_factors), zne_evs[:,j] , 0)
            extrapolate_exp[:,j] = coeff
            extrapolate_value = extrapolate_exp

    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Function executed in: {execution_time:.6f} seconds")
    # return extrapolation
    return {"extrapolated_value": extrapolate_value.tolist(),
           "job_runtime": execution_time}  # ← .tolist() for JSON

