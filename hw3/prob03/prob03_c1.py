
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

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


from qiskit_nature.second_q.circuit.library import UCC, HartreeFock

# JW
hf_jw = HartreeFock(
    num_spatial_orbitals=2,
    num_particles=(1, 1),
    qubit_mapper=jw_mapper,
)
uccsd_jw = UCC(
    num_spatial_orbitals=2,
    num_particles=(1, 1),
    qubit_mapper=jw_mapper,
    initial_state=hf_jw,
    reps=1,
    excitations="d"
)
print(uccsd_jw.decompose().draw("text"))
uccsd_jw.decompose().draw(output="mpl", filename="uccsd_jw_circuit.png")

# Parity-Tapered 
hf_par = HartreeFock(
    num_spatial_orbitals=2,
    num_particles=(1, 1),
    qubit_mapper=par_mapper,
)
uccsd_par = UCC(
    num_spatial_orbitals=2,
    num_particles=(1, 1),
    qubit_mapper=par_mapper,
    initial_state=hf_par,
    reps=1,
    excitations="d"
)
print(uccsd_par.decompose().draw("text"))
uccsd_par.decompose().draw(output="mpl", filename="uccsd_parity_circuit.png")
