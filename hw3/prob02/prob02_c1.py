import warnings
warnings.filterwarnings("ignore")

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

print(f"\n{'Eigenvalue':<13}  {'JW (4q)':>14}  {'BK (4q)':>14}  {'Parity (2q)':>16}")
print("-"*75)
for i in range(4):
    print(f"  E{i}  (Ha)     {evals_jw[i]:>14.6f}  "
          f"{evals_bk[i]:>14.6f}  {evals_par[i]:>16.6f}")
