import numpy as np
import dimod
import time

seed = 10608068        
np.random.seed(seed)

weights = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
values  = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
W = 165

best_value = 309

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

exact_results = {}

print("┌────────┬────────────┬───────┬───────┬──────────┬─────────┬──────────┐")
print("│   λ    │    bits    │   w   │   v   │ feasible │ optimal │   time   │")
print("├────────┼────────────┼───────┼───────┼──────────┼─────────┼──────────┤")
for lambd in [0.07, 0.08, 1.0, 4.0, 10.0]:
    Q, N, M = build_qubo(weights, values, W, lambd)
    bqm = dimod.BQM.from_qubo(Q)
    t0 = time.perf_counter()
    res = dimod.ExactSolver().sample(bqm)
    t_e = time.perf_counter() - t0
    samp = res.first.sample
    bits = ''.join(str(samp[i]) for i in range(len(weights)))
    tw = sum(weights[i] for i, c in enumerate(bits) if c == '1')
    tv = sum(values[i]  for i, c in enumerate(bits) if c == '1')
    feas = tw <= W
    print(f"│ {lambd:>6.2f} │ {bits:<10} │ {tw:>5d} │ {tv:>5d} │ {str(feas):>8} │ {str(feas and tv==best_value):>7} │ {t_e*1000:>6.1f}ms │")

print("└────────┴────────────┴───────┴───────┴──────────┴─────────┴──────────┘")