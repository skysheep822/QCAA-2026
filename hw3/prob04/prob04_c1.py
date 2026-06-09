
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

### main ###
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeNairobiV2 

SEED = 10608068
np.random.seed(SEED)

# Noise model
fake_backend = FakeNairobiV2()
noise_model  = NoiseModel.from_backend(fake_backend)

theta_opt_val = -0.11306814 # from prob04_a1.py UCCSD+JW
bound_par = uccsd_par.assign_parameters({uccsd_par.parameters[0]: theta_opt_val})

def shot_estimate(circuit, hamiltonian, n_shots, nm):
    np.random.seed(SEED)
    est = AerEstimator()
    if nm is not None:
        est.set_options(shots=n_shots, seed=SEED, noise_model=nm)
    else:
        est.set_options(shots=n_shots, seed=SEED)

    job    = est.run([circuit], [hamiltonian])
    E_mean = job.result().values[0]
    se = np.sqrt(job.result().metadata[0]["variance"])
    print(f"{n_shots},{E_mean:.6f},{se:.6f}")

print(f"\nNo_Noise")
shot_estimate(bound_par, H_par, 10, None)
shot_estimate(bound_par, H_par, 10000, None)

print(f"\nFakeNairobi_Noise")
shot_estimate(bound_par, H_par, 10, noise_model)
shot_estimate(bound_par, H_par, 10000, noise_model)
