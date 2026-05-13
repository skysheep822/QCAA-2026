def autocorrelation(s, k):
    N = len(s)
    return sum(s[i] * s[i + k] for i in range(N - k))

def sidelobe_energy(s):
    N = len(s)
    return sum(autocorrelation(s, k) ** 2 for k in range(1, N))

def merit_factor(s):
    N = len(s)
    E = sidelobe_energy(s)
    if E == 0: return float('inf')
    return (N ** 2) / (2 * E)

barker_str = "+++---+--+-"
barker_en  = [1 if c == '+' else -1 for c in barker_str]
print(f"E = {sidelobe_energy(barker_en)}, F = {merit_factor(barker_en)}")

import time
import numpy as np
for n in [5, 11, 13, 20]:
    best_E, best_F, best_seq = float('inf'), 0, []
    t0 = time.time()
    for case in range(2**n):
        bits = np.binary_repr(case, n)
        seq = [1 if c == '1' else -1 for c in bits]
        E = sidelobe_energy(seq)
        F = merit_factor(seq)
        if E < best_E:
            best_E = E
            best_F = F
            best_seq = seq
    t_elapsed = time.time() - t0
    best_seq_str = ''.join('+' if x == 1 else '-' for x in best_seq)
    print(f"N: {n}, {best_seq_str}, E = {best_E}, F = {best_F}, elapsed = {t_elapsed:.4f} 秒")
