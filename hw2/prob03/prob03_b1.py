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

def build_labs_hamiltonian(N):
    ops, coeffs = [], []
    for k in range(1, N):
        for i in range(N - k):
            for j in range(N - k):
                q_list = [i, i + k, j, j + k]
                counts  = Counter(q_list)
                effective = sorted(q for q, cnt in counts.items() if cnt % 2 == 1)
                if not effective:
                    continue   
                ops.append(qml.prod(*[qml.PauliZ(q) for q in effective]))
                coeffs.append(1.0)
    return qml.Hamiltonian(coeffs, ops)

cost_h = build_labs_hamiltonian(N_seq)
print(f"HC 項數: {len(cost_h.ops)}")

wires = range(N_seq)
dev = qml.device("default.qubit", wires=wires)

def qaoa_layer(gamma, beta):
    for coeff, op in zip(cost_h.coeffs, cost_h.ops):
        wire_list = list(op.wires)
        qml.PauliRot(2.0 * float(coeff) * gamma, "Z" * len(wire_list), wires=wire_list)
    for w in wires:
        qml.RX(2.0 * beta, wires=w)

def circuit(params, depth, **kwargs):
    for w in wires:
        qml.Hadamard(wires=w)
    qml.layer(qaoa_layer, depth, params[0], params[1])

@qml.qnode(dev)
def cost_function(params, depth=1):
    circuit(params, depth=depth)
    return qml.expval(cost_h)

def interp_params(params_p):
    """INTERP: 將深度 p 的最優參數插值成深度 p+1 """
    g_p, b_p = params_p[0], params_p[1]
    p = len(g_p)
    g_new = np.zeros(p + 1)
    b_new = np.zeros(p + 1)
    for i in range(1, p + 2):   # 1-indexed
        if i == 1:
            g_new[0] = (p - i + 1) / p * g_p[0]
            b_new[0] = (p - i + 1) / p * b_p[0]
        elif i == p + 1:
            g_new[p] = (i - 1) / p * g_p[p - 1]
            b_new[p] = (i - 1) / p * b_p[p - 1]
        else:
            g_new[i-1] = (i-1)/p * g_p[i-2] + (p-i+1)/p * g_p[i-1]
            b_new[i-1] = (i-1)/p * b_p[i-2] + (p-i+1)/p * b_p[i-1]
    return np.array([g_new, b_new])

# greedy-MLI 策略（arxiv-2306-06986）:
#   p=1 用 N_INIT_P1 個隨機起始點平行優化，保留最佳 K_GREEDY 個
#   p>1 對所有保留的 K 個參數集做 INTERP，各自優化後再保留最佳 K 個
#   最終取各層最佳解回報
K_GREEDY  = 3   # 每層保留最佳 K 個參數集
N_INIT_P1 = 5   # p=1 的隨機起始點數

def optimize_multi(depth, init_list):
    """對多個初始點分別跑 COBYLA, 回傳按 cost 由小到大排序的 [(params_2d, val)] """
    def _cost_fn(flat_params):
        gamma = flat_params[:depth]
        beta  = flat_params[depth:]
        return float(cost_function(np.array([gamma, beta]), depth=depth))

    results = []
    for x0 in init_list:
        res = minimize(_cost_fn, x0, method="COBYLA",
                       options={"maxiter": 300 * depth, "rhobeg": 0.5})
        opt = res.x
        results.append((np.array([opt[:depth], opt[depth:]]), res.fun))
    results.sort(key=lambda x: x[1])
    return results   # [(params_2d, val), ...]

@qml.qnode(dev)
def probability_circuit(gamma, alpha, depth=1):
    circuit([gamma, alpha], depth=depth)
    return qml.probs(wires=wires)

rng = np.random.default_rng(seed)
top_params = None   # List[ndarray shape (2, depth)]，每層保留最佳 K_GREEDY 個

print(f"p, best spin, E, F, approx ratio, time, γ, β:")
for depth in [1, 2, 3]:
    t0 = time.time()

    if depth == 1:
        # p=1：多起始點隨機初始化（greedy-MLI 的關鍵）
        init_list = [rng.uniform(0, np.pi, 2) for _ in range(N_INIT_P1)]
    else:
        # p>1：對所有保留的 K 個參數集各做 INTERP，生成對應候選起始點
        # 不再加入隨機點——greedy-MLI 在 p>1 完全依賴 INTERP 繼承先驗知識
        init_list = []
        for prev in top_params:
            seeded = interp_params(prev)
            init_list.append(np.concatenate([seeded[0], seeded[1]]))

    results    = optimize_multi(depth, init_list)
    top_params = [r[0] for r in results[:K_GREEDY]]
    best_params = top_params[0]   # 本層最優解
    elapsed = time.time() - t0

    probs    = probability_circuit(best_params[0], best_params[1], depth=depth)
    best_idx = int(np.argmax(probs))

    best_bits = [(best_idx >> (N_seq - 1 - b)) & 1 for b in range(N_seq)]
    best_spin = [1 - 2 * b for b in best_bits]
    best_str  = ''.join('+' if x == 1 else '-' for x in best_spin)

    E_best       = sidelobe_energy(best_spin)
    F_best       = merit_factor(best_spin)
    approx_ratio = F_best / F_star


    print(f"{depth}, {best_str}, {E_best:.4f}, {F_best:.4f}, "
          f"{approx_ratio:.4f}, "
          f"{elapsed:.2f}s", 
          f"{best_params[0]}, {best_params[1]}")
