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

def build_labs_qubo(N):
    # 收集 s_i s_{i+k} s_j s_{j+k} 中 4 個 index 恰好只有 2 個不同值的項
    # 這些項化簡後為 2-local Ising 交互 J_{ab}
    Q = {}
    for k in range(1, N):
        for i in range(N - k):
            for j in range(N - k):
                idx_set = {i, i + k, j, j + k}
                if len(idx_set) == 2:
                    a, b = sorted(idx_set)
                    Q[(a, b)] = Q.get((a, b), 0) + 1.0
    return Q

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

N_INIT_P1 = 5

def optimize_multi(depth, init_list):
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
    return results

rng = np.random.default_rng(seed)

t0 = time.time()
init_list = [rng.uniform(0, np.pi, 2) for _ in range(N_INIT_P1)]
results_p1 = optimize_multi(1, init_list)
best_params_p1 = results_p1[0][0]
print(f"p=1 optimized, cost={results_p1[0][1]:.4f}, time={time.time()-t0:.2f}s, "
      f"γ={best_params_p1[0]}, β={best_params_p1[1]}")

dev_sample = qml.device("default.qubit", wires=wires, shots=200)

@qml.qnode(dev_sample)
def sample_circuit(params):
    circuit(params, depth=1)
    return qml.sample(wires=wires)

quantum_bits = sample_circuit(best_params_p1)
quantum_spin = 1 - 2 * quantum_bits

sample_energies = [sidelobe_energy(list(q)) for q in quantum_spin]
sorted_idx = np.argsort(sample_energies)
top_seeds = quantum_spin[sorted_idx[:10]]

print(f"min E: {min(sample_energies)}")

def local_search_labs(s_init, max_iter=300, tabu_tenure_min=5, tabu_tenure_max=10):
    s = list(s_init)
    best_s = s[:]
    best_E = sidelobe_energy(s)
    tabu = {}
    n_iter = 0

    for it in range(max_iter):
        n_iter += 1
        best_nb_E = float('inf')
        best_flip = -1

        for i in range(len(s)):
            if tabu.get(i, 0) > it:
                continue
            s[i] = -s[i]
            E_new = sidelobe_energy(s)
            s[i] = -s[i]
            if E_new < best_nb_E:
                best_nb_E = E_new
                best_flip = i

        if best_flip == -1:
            break

        tenure = tabu_tenure_min + int((tabu_tenure_max - tabu_tenure_min) * it / max_iter)
        s[best_flip] = -s[best_flip]
        tabu[best_flip] = it + tenure
        if best_nb_E < best_E:
            best_E = best_nb_E
            best_s = s[:]

    return best_s, best_E, n_iter

best_E = float('inf')
best_seq = ()
n_eval_local = 0

t0 = time.time()
for seed_seq in top_seeds:
    t1 = time.time()
    refined_seq, refined_E, iters = local_search_labs(list(seed_seq), max_iter=300)
    n_eval_local += iters
    if refined_E < best_E:
        best_E = refined_E
        best_seq = refined_seq
    
    print(f"Local search from {''.join('+' if x == 1 else '-' for x in seed_seq)}: "
          f"E={refined_E}, iters={iters}, time={time.time()-t1:.2f}s")

elased = time.time() - t0
best_F = merit_factor(best_seq)
r = best_F / F_star
best_str = ''.join('+' if x == 1 else '-' for x in best_seq)
n_eval_total = 200 + n_eval_local

print(f"\n結果:")
print(f"{best_str}, "
      f"E={best_E}, F={best_F:.4f}, r={r:.4f}, N_eval={n_eval_total}, time={elased:.2f}s")