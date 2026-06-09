import warnings
warnings.filterwarnings("ignore")

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    BravyiKitaevMapper,
    ParityMapper,
)
from qiskit_nature.units import DistanceUnit

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

def print_hamiltonian(name, op):
    print(f"\n{name} ({op.num_qubits} qubits, {len(op)} terms)")
    for pauli, coeff in sorted(op.label_iter(), key=lambda x: x[0]):
        if abs(coeff) > 1e-10:
            print(f"{pauli},{coeff.real:+.8f}")

print_hamiltonian("JW Hamiltonian",           H_jw)
print_hamiltonian("BK Hamiltonian",           H_bk)
print_hamiltonian("Parity-Tapered (2-qubit)", H_par)
