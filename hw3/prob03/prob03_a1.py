import warnings
warnings.filterwarnings("ignore")
import numpy as np

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
problem = PySCFDriver(
    atom="H 0 0 0; H 0 0 0.7414",
    basis="sto3g",
    unit=DistanceUnit.ANGSTROM,
    charge=0,
    spin=0,
).run()

# one-electron integrals hpq
print("\none-electron integrals h[p,q]:")
hamiltonian = problem.hamiltonian
h_mo = hamiltonian.electronic_integrals.one_body.alpha["+-"]
print(f"h[1,1] = {h_mo[0,0]:+.10f}")
print(f"h[1,2] = {h_mo[0,1]:+.10f}")
print(f"h[2,1] = {h_mo[1,0]:+.10f}")
print(f"h[2,2] = {h_mo[1,1]:+.10f}")

# six symmetry-unique electron-repulsion integrals
print("\nsix symmetry-unique two-electron integrals:")
n = h_mo.shape[0]
eri_chem = np.zeros((n, n, n, n))
for val, idx in hamiltonian.electronic_integrals.two_body.alpha["++--"].coord_iter():
    eri_chem[idx] = val

uniq = [
    (0, 0, 0, 0, "<11|11>"),
    (0, 0, 0, 1, "<11|12>"),
    (0, 0, 1, 1, "<11|22>"),
    (0, 1, 0, 1, "<12|12>"),
    (0, 1, 1, 1, "<12|22>"),
    (1, 1, 1, 1, "<22|22>"),
]
def dirac(p, q, r, s):
    return float(eri_chem[p, r, q, s]) 
for p, q, r, s, label in uniq:
    print(f"{label} = {dirac(p,q,r,s):+.10f}")

# E_nuc
E_nuc = hamiltonian.nuclear_repulsion_energy
print(f"\nE_nuc  = {E_nuc:.10f}  Ha")

