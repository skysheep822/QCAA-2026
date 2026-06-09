import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from functools import partial

SEED = 10608068   

np.random.seed(SEED)

# pyscf
from pyscf import cc, fci, gto, scf, ao2mo

mol_lih = gto.M(
    atom="Li 0 0 0; H 0 0 1.5957",
    basis="sto-3g", unit="Angstrom",
    charge=0, spin=0, verbose=0,
)
mf_lih = scf.RHF(mol_lih).run()
E_nuc_lih  = mol_lih.energy_nuc()
E_HF_lih   = mf_lih.e_tot
num_orb    = mol_lih.nao_nr()   
n_alpha = n_beta = 2
nelec    = (n_alpha, n_beta)

C_lih    = mf_lih.mo_coeff
hcore_lih = C_lih.T @ mf_lih.get_hcore() @ C_lih
eri_lih   = ao2mo.restore(1, ao2mo.kernel(mol_lih, C_lih), num_orb)

mycc      = cc.CCSD(mf_lih).run()
t1_ccsd   = mycc.t1
t2_ccsd   = mycc.t2
E_CCSD_lih = mycc.e_tot

myfci = fci.FCI(mf_lih)
myfci.verbose = 0
E_FCI_lih, _ = myfci.kernel()

print(f"\n  E_nuc   = {E_nuc_lih:.5f}  Ha")
print(f"  E_HF    = {E_HF_lih:.5f}  Ha")
print(f"  E_CCSD  = {E_CCSD_lih:.5f}  Ha")
print(f"  E_FCI   = {E_FCI_lih:.5f}  Ha")

# ── UCCSD 電路（12 qubits，JW blocked ordering）─────────────────
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper
)

driver_lih = PySCFDriver(
    atom="Li 0 0 0; H 0 0 1.5957",
    basis="sto-3g", unit=DistanceUnit.ANGSTROM,
    charge=0, spin=0,
)
problem_lih  = driver_lih.run()
mapper_lih   = JordanWignerMapper()

hf_lih_circ = HartreeFock(
    num_spatial_orbitals=num_orb,
    num_particles=(n_alpha, n_beta),
    qubit_mapper=mapper_lih,
)
uccsd_lih = UCCSD(
    num_spatial_orbitals=num_orb,
    num_particles=(n_alpha, n_beta),
    qubit_mapper=mapper_lih,
    initial_state=hf_lih_circ,
    reps=1,
)
n_qubits_lih = uccsd_lih.num_qubits  


# classical CCSD + pyscf
def ccsd_to_uccsd_params(uccsd_circ, t1, t2, num_orb, nocc):
    params = np.zeros(uccsd_circ.num_parameters)
    exc    = uccsd_circ.excitation_list
    for idx, (occ_modes, vir_modes) in enumerate(exc):
        if len(occ_modes) == 1:
            i_sp = occ_modes[0] % num_orb
            a_sp = vir_modes[0] % num_orb
            if i_sp < nocc and nocc <= a_sp:
                params[idx] = t1[i_sp, a_sp - nocc]
        elif len(occ_modes) == 2:
            i_mode, j_mode = occ_modes
            a_mode, b_mode = vir_modes
            i_sp, j_sp = i_mode % num_orb, j_mode % num_orb
            a_sp, b_sp = a_mode % num_orb, b_mode % num_orb
            i_is_a = (i_mode < num_orb); j_is_a = (j_mode < num_orb)
            if i_sp < nocc and j_sp < nocc and nocc <= a_sp and nocc <= b_sp:
                av, bv = a_sp - nocc, b_sp - nocc
                if i_is_a and not j_is_a:
                    params[idx] = t2[i_sp, j_sp, av, bv]
                elif not i_is_a and j_is_a:
                    params[idx] = t2[j_sp, i_sp, bv, av]
                else:
                    params[idx] = t2[i_sp, j_sp, av, bv] - t2[i_sp, j_sp, bv, av]
    return params

init_params_lih = ccsd_to_uccsd_params(uccsd_lih, t1_ccsd, t2_ccsd, num_orb, n_alpha)

# qiskit_addon_sqd
from qiskit.primitives import StatevectorSampler
from qiskit_addon_sqd.fermion import (
    diagonalize_fermionic_hamiltonian,
    solve_sci_batch,
)

uccsd_lih_meas = uccsd_lih.copy()
uccsd_lih_meas.measure_all()

sampler_lih = StatevectorSampler(seed=np.random.default_rng(SEED))

sci_solver = partial(solve_sci_batch, spin_sq=0.0, max_cycle=200)

SHOT_BUDGETS          = [100, 1_000, 10_000, 100_000]
SAMPLES_PER_BATCH_MAP = {100: 50, 1_000: 200, 10_000: 500, 100_000: 1_000}
# SHOT_BUDGETS          = [10]
# SAMPLES_PER_BATCH_MAP = {10: 25}
NUM_BATCHES = 2

results_sqd = []

for n_shots in SHOT_BUDGETS:
    print(f"\nShot: {n_shots}")
    pub = (uccsd_lih_meas, init_params_lih.reshape(1, -1), n_shots)
    bit_array = sampler_lih.run([pub]).result()[0].data.meas[0]

    sqd_res = diagonalize_fermionic_hamiltonian(
        hcore_lih,
        eri_lih,
        bit_array,
        samples_per_batch   = SAMPLES_PER_BATCH_MAP[n_shots],
        norb                = num_orb,
        nelec               = nelec,
        num_batches         = NUM_BATCHES,
        energy_tol          = 1e-4,
        occupancies_tol     = 1e-4,
        max_iterations      = 10,
        sci_solver          = sci_solver,
        symmetrize_spin     = True,
        carryover_threshold = 1e-4,
        seed                = np.random.default_rng(SEED + n_shots),
    )

    sci_state = sqd_res.sci_state
    d_alpha   = len(sci_state.ci_strs_a)
    d_ci      = len(sci_state.ci_strs_a) * len(sci_state.ci_strs_b)
    E_SQD     = sqd_res.energy + E_nuc_lih
    abs_err   = abs(E_SQD - E_FCI_lih)
    results_sqd.append((n_shots, d_alpha, d_ci, E_SQD, abs_err))

    print(f"d_alpha: {d_alpha}, d_ci: {d_ci}")
    print(f"E_SQD: {E_SQD:.6f}, abs_err: {abs_err:.4e}")

# ── Problem 5(ii)：收斂圖 ────────────────────────────────────────
shots_list  = [r[0] for r in results_sqd]
d_ci_list   = [r[2] for r in results_sqd]
err_list    = [r[4] for r in results_sqd]
CHEM_ACC    = 1.6e-3

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("SQD Convergence: LiH STO-3G (4e/6o), UCCSD+CCSD init, JW 12-qubit",
             fontsize=12, fontweight="bold")

ax = axes[0]
ax.plot(d_ci_list, err_list, "o-b", lw=2, ms=8, label="|ΔE|")
ax.axhline(CHEM_ACC, color="r", ls="--", lw=1.5, label="Chem. accuracy (1.6 mHa)")
ax.axhline(abs(E_CCSD_lih - E_FCI_lih), color="orange", ls=":", lw=1.5,
           label=f"CCSD error ({abs(E_CCSD_lih - E_FCI_lih):.2e} Ha)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("CI Subspace Dimension $d^2$")
ax.set_ylabel(r"$|E_\mathrm{SQD}-E_\mathrm{FCI}|$ (Ha)")
ax.set_title("vs. Subspace Dimension"); ax.legend(); ax.grid(True, alpha=0.3)
for x, y, n in zip(d_ci_list, err_list, shots_list):
    ax.annotate(f"{n}", (x, y), xytext=(5, 6), textcoords="offset points", fontsize=9)

ax = axes[1]
ax.plot(shots_list, err_list, "s-g", lw=2, ms=8, label="|ΔE|")
ax.axhline(CHEM_ACC, color="r", ls="--", lw=1.5, label="Chem. accuracy")
ax.axhline(abs(E_CCSD_lih - E_FCI_lih), color="orange", ls=":", lw=1.5)
n_ref = np.array([1e2, 1e5])
scale = err_list[0] * np.sqrt(shots_list[0])
ax.plot(n_ref, scale / np.sqrt(n_ref), "k--", alpha=0.4, label=r"$\propto 1/\sqrt{N}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Shot Count $N_\\mathrm{shots}$")
ax.set_ylabel(r"$|E_\mathrm{SQD}-E_\mathrm{FCI}|$ (Ha)")
ax.set_title("vs. Shot Count"); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sqd_lih_convergence.png", dpi=150, bbox_inches="tight")