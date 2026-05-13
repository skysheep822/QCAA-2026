import numpy as np
import dimod
import neal
import time

seed = 0        
np.random.seed(seed)

weights = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
values  = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
W = 165
n = len(weights)

def build_qubo(weights, values, max_weight, lambd):
    M = int(np.ceil(np.log2(max_weight)))
    w_ext = weights + [2**k for k in range(M)]
    v_ext = list(values) + [0] * M
    N = len(weights) + M
    Q = {}
    for i in range(N):
        Q[(i, i)] = -v_ext[i] + lambd * w_ext[i] * (w_ext[i] - 2 * max_weight)
        for j in range(i + 1, N):
            Q[(i, j)] = 2 * lambd * w_ext[i] * w_ext[j]
    return Q, N, M

best_bits = "1111010000"
best_value = 309
best_lambd = 4.0
Q, N, M = build_qubo(weights, values, W, best_lambd)
bqm = dimod.BQM.from_qubo(Q)

num_reads_list = [10, 100, 1000, 10000]
print("┌──────────────┬───────────────────────┬───────────┐")
print("│  num_reads   │     success rate      │   time    │")
print("├──────────────┼───────────────────────┼───────────┤")
for nr in num_reads_list:
    t0 = time.perf_counter()
    res = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=nr, seed=seed)
    t_sa = time.perf_counter() - t0
    success = sum(
        occ for samp, _, occ in res.data(['sample', 'energy', 'num_occurrences'])
        if ''.join(str(samp[i]) for i in range(n)) == best_bits
    )
    prob = success / nr
    print(f"│ {nr:>12d} │ {success:>5d}/{nr:<6d} ({prob*100:5.1f}%) │ {t_sa*1000:>7.1f}ms │")
print("└──────────────┴───────────────────────┴───────────┘")


# Fine-grained lambda search
lambdas = [0.03, 0.05, 0.076, 0.08, 0.1, 0.5, 1.0, 4.0, 10.0]
print("┌────────┬──────────────────────┬───────────┐")
print("│   λ    │     success rate     │   time    │")
print("├────────┼──────────────────────┼───────────┤")
for lambd in lambdas:
    Q, N, M = build_qubo(weights, values, W, lambd)
    bqm_l = dimod.BQM.from_qubo(Q)
    t0 = time.perf_counter()
    res = neal.SimulatedAnnealingSampler().sample(bqm_l, num_reads=10000, seed=seed)
    t_sa = time.perf_counter() - t0
    success = sum(
        occ for samp, _, occ in res.data(['sample', 'energy', 'num_occurrences'])
        if ''.join(str(samp[i]) for i in range(n)) == best_bits
    )
    prob = success / 10000
    print(f"│ {lambd:>6.3f} │ {success:>5d}/10000 ({prob*100:5.1f}%) │ {t_sa*1000:>7.1f}ms │")
print("└────────┴──────────────────────┴───────────┘")

