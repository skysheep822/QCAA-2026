import time
from collections import Counter

import neal
import dimod
import numpy as np
import pennylane as qml
from scipy.optimize import minimize

STAR = {
    5:  (2, 6.25),
    11: (5, 12.10),
    13: (6, 14.08),
    20: (26, 7.69),
}

N_seq  = 20     # LABS
seed   = 10608068
E_opt, F_star = STAR[N_seq]

print(f"LABS N={N_seq}, E_opt={E_opt}, F*={F_star}")

def sidelobe_energy(s):
    N = len(s)
    E = 0
    for k in range(1, N):
        ck = sum(s[i] * s[i + k] for i in range(N - k))
        E += ck ** 2
    return E

def merit_factor(s):
    N = len(s)
    E = sidelobe_energy(s)
    return (N ** 2) / (2 * E) if E > 0 else float('inf')

# Pure SA baseline
def labs_to_ising_bqm(N):
    """近似：只取 2-local 交叉項作為 Ising 模型（完整 4-local 轉換省略）。"""
    # 此處以 'SPIN' 變數型態建立 BQM
    linear = {}
    quadratic = {}
    # 遍歷所有 k, i, j 計算 J_{mn} 係數（合計所有貢獻）
    J = np.zeros((N, N))
    for k in range(1, N):
        for i in range(N - k):
            for j in range(N - k):
                # s_i s_{i+k} s_j s_{j+k}
                # 當兩對不重疊時純 4-local；重疊時含 2-local 貢獻
                idx_set = {i, i+k, j, j+k}
                if len(idx_set) == 2:
                    a, b = sorted(idx_set)
                    J[a][b] += 1.0
                # 4-local 項在 2-local BQM 中無法精確表示；此為粗化近似
    for a in range(N):
        for b in range(a+1, N):
            if J[a][b] != 0:
                quadratic[(a, b)] = J[a][b]

    bqm = dimod.BQM(linear, quadratic, offset=0, vartype='SPIN')
    return bqm

bqm_labs = labs_to_ising_bqm(N_seq)

sa_sampler = neal.SimulatedAnnealingSampler()
t_start = time.time()
sa_result = sa_sampler.sample(bqm_labs, num_reads=1000, seed=seed)
t_sa = time.time() - t_start

best_sa_sample = sa_result.first.sample
best_sa_spin   = [int(best_sa_sample[i]) for i in range(N_seq)]
best_sa_E      = sidelobe_energy(best_sa_spin)
best_sa_F      = (N_seq ** 2) / (2 * best_sa_E) if best_sa_E > 0 else float('inf')
r_sa           = best_sa_F / F_star
print(f"Pure SA: E_best={best_sa_E}, F_best={best_sa_F:.4f}, r={r_sa:.4f}, time: {t_sa:.2f}s")