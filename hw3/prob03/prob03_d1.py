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

from qiskit.circuit.library import RealAmplitudes
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit import QuantumCircuit

# JW
hf_jw = HartreeFock(
    num_spatial_orbitals=2,    
    num_particles=(1, 1),      
    qubit_mapper=jw_mapper,
)

# ansatz
ra_jw_circ = RealAmplitudes(
    num_qubits=4,
    reps=1,
    entanglement="linear",
    insert_barriers=True,
    initial_state=hf_jw,  
)
print(ra_jw_circ.decompose().draw("text"))
ra_jw_circ.decompose().draw(output="mpl", filename="ra_jw_circuit.png")

# Parity-Tapered 
hf_par = HartreeFock(
    num_spatial_orbitals=2,
    num_particles=(1, 1),
    qubit_mapper=par_mapper,
)

# ansatz
ra_par_circ = RealAmplitudes(
    num_qubits=2,
    reps=1,
    entanglement="linear",
    insert_barriers=True,
    initial_state=hf_par,  
)
print(ra_par_circ.decompose().draw("text"))
ra_par_circ.decompose().draw(output="mpl", filename="ra_parity_circuit.png")