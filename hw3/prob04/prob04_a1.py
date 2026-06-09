
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

### Molecular Hamiltonian ###
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    BravyiKitaevMapper,
    ParityMapper,
)
from qiskit_nature.units import DistanceUnit
import numpy as np

problem = PySCFDriver(
    atom="H 0 0 0; H 0 0 0.7414",
    basis="sto3g",
    unit=DistanceUnit.ANGSTROM,
    charge=0,
    spin=0,
).run()
ham_second_q = problem.hamiltonian.second_q_op()

jw_mapper  = JordanWignerMapper()
bk_mapper  = BravyiKitaevMapper()
par_mapper = ParityMapper(num_particles=(1, 1))  

H_jw  = jw_mapper.map(ham_second_q) 
H_bk  = bk_mapper.map(ham_second_q) 
H_par = par_mapper.map(ham_second_q)

def lowest_evals(spop, k=4):
    mat = spop.to_matrix()
    evals = np.sort(np.linalg.eigvalsh(mat))
    return evals[:k]

evals_jw  = lowest_evals(H_jw)
evals_bk  = lowest_evals(H_bk)
evals_par = lowest_evals(H_par)

### HF ###
from qiskit_nature.second_q.circuit.library import HartreeFock
hf_jw = HartreeFock(
    num_spatial_orbitals=2,
    num_particles=(1, 1),
    qubit_mapper=jw_mapper,
)
hf_par = HartreeFock(
    num_spatial_orbitals=2,
    num_particles=(1, 1),
    qubit_mapper=par_mapper,
)

### UCCSD ansatz ###
from qiskit_nature.second_q.circuit.library import UCC

# JW
uccsd_jw = UCC(
    num_spatial_orbitals=2,
    num_particles=(1, 1),
    qubit_mapper=jw_mapper,
    initial_state=hf_jw,
    reps=1,
    excitations="d"
)

# Parity-Tapered 
uccsd_par = UCC(
    num_spatial_orbitals=2,
    num_particles=(1, 1),
    qubit_mapper=par_mapper,
    initial_state=hf_par,
    reps=1,
    excitations="d"
)

### real-amplitude ansatz ###
from qiskit.circuit.library import RealAmplitudes

# JW
ra_jw_circ = RealAmplitudes(
    num_qubits=4,
    reps=1,
    entanglement="linear",
    insert_barriers=True,
    initial_state=hf_jw,  
)

# Parity-Tapered 
ra_par_circ = RealAmplitudes(
    num_qubits=2,
    reps=1,
    entanglement="linear",
    insert_barriers=True,
    initial_state=hf_par,  
)


### main ###

from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

SEED = 10608068
np.random.seed(SEED)

def run_vqe(hamiltonian, ansatz_circ, label):
    np.random.seed(SEED)
    estimator = StatevectorEstimator()
    opt = COBYLA(maxiter=1000, tol=1e-8, rhobeg=1.0)
    n_params = ansatz_circ.num_parameters

    rng = np.random.default_rng(SEED)
    initial_point = rng.uniform(-0.1, 0.1, n_params)

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz_circ,
        optimizer=opt,
        initial_point=initial_point,
    )
    result   = vqe.compute_minimum_eigenvalue(hamiltonian)
    E_opt    = result.eigenvalue.real
    params   = result.optimal_point
    n_evals  = result.cost_function_evals

    print(f"{label},{E_opt:.8f},{params},{n_params},{n_evals}")

run_vqe(H_jw,  uccsd_jw,   "UCCSD+JW")
run_vqe(H_par, uccsd_par,  "UCCSD+Parity")
run_vqe(H_jw,  ra_jw_circ, "RA+JW")
run_vqe(H_par, ra_par_circ,"RA+Parity")