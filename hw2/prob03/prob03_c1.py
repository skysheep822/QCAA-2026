import time
from collections import Counter

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


# 1. Random sampling baseline
np.random.seed(seed)
N_random_shots = 10000
best_random_E = float('inf')
best_random_seq = None

t0 = time.time()
for _ in range(N_random_shots):
    s_rand = np.random.choice([-1, 1], size=N_seq)
    E_rand = sidelobe_energy(s_rand)
    if E_rand < best_random_E:
        best_random_E = E_rand
        best_random_seq = s_rand
eslapsed = time.time() - t0

F_random_best = (N_seq ** 2) / (2 * best_random_E)
r_random = F_random_best / F_star
print(f"Random ({N_random_shots} shots): E_best={best_random_E}, F_best={F_random_best:.4f}, r={r_random:.4f}, time: {eslapsed:.2f}s")
