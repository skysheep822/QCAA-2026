from collections import Counter

import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

N_seq = 5
seed  = 10608068

STAR = {5: (2, 6.25), 11: (5, 12.10), 13: (6, 14.08), 20: (26, 7.69)}
E_opt, F_star = STAR[N_seq]

def sidelobe_energy(s):
    N = len(s)
    E = 0
    for k in range(1, N):
        ck = sum(s[i] * s[i + k] for i in range(N - k))
        E += ck ** 2
    return E

def build_labs_hamiltonian(N):
    ops, coeffs = [], []
    for k in range(1, N):
        for i in range(N - k):
            for j in range(N - k):
                q_list = [i, i + k, j, j + k]
                counts = Counter(q_list)
                effective = sorted(q for q, cnt in counts.items() if cnt % 2 == 1)
                if not effective:
                    continue
                ops.append(qml.prod(*[qml.PauliZ(q) for q in effective]))
                coeffs.append(1.0)
    return qml.Hamiltonian(coeffs, ops)

cost_h = build_labs_hamiltonian(N_seq)
wires  = range(N_seq)
dev    = qml.device("default.qubit", wires=wires)

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

gammas = np.linspace(0, 2 * np.pi, 50)
betas  = np.linspace(0,     np.pi, 50)
landscape = np.zeros((len(gammas), len(betas)))

for i, g in enumerate(gammas):
    for j, b in enumerate(betas):
        landscape[i, j] = cost_function(np.array([[g], [b]]))

plt.figure(figsize=(7, 5))
plt.imshow(landscape, origin="lower",
           extent=[0, np.pi, 0, 2 * np.pi], aspect="auto", cmap="RdYlBu_r")
plt.colorbar(label=r"$\langle H_C \rangle$")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$\gamma$")
plt.title(f"QAOA p=1 landscape — LABS N={N_seq}")
plt.tight_layout()
plt.savefig("labs_landscape_n5.png", dpi=150)
plt.show()

min_idx    = np.unravel_index(np.argmin(landscape), landscape.shape)
best_gamma = gammas[min_idx[0]]
best_beta  = betas[min_idx[1]]
print(f"Global minimum: γ={best_gamma:.4f}, β={best_beta:.4f}, "
      f"⟨H_C⟩={landscape[min_idx]:.4f}")
