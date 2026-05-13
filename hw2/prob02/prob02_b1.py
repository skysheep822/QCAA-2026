import networkx as nx
import numpy as np
import pennylane as qp
import matplotlib.pyplot as plt
from pennylane import qaoa
from sympy import im    

# gen random graph
seed = 10608068 
np.random.seed(seed)
G = nx.gnp_random_graph(n=8, p=0.5, seed=seed)
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
print(f"Edges: {list(G.edges())}")

# QAOA
wires = range(G.number_of_nodes())
cost_h, mixer_h = qaoa.maxcut(G)

def qaoa_layer(gamma, alpha):
    qaoa.cost_layer(gamma, cost_h)
    qaoa.mixer_layer(alpha, mixer_h)

def circuit(params, depth,  **kwargs):
    for w in wires:
        qp.Hadamard(wires=w)
    qp.layer(qaoa_layer, depth, params[0], params[1])   # depth=1 for landscape

dev = qp.device("default.qubit", wires=wires)
@qp.qnode(dev)
def cost_function(params, depth=1):
    circuit(params, depth=depth)
    return qp.expval(cost_h)

gammas = np.linspace(0, 2 * np.pi, 50)
betas  = np.linspace(0,     np.pi, 50)
parms = np.array([[gammas[0]], [betas[0]]]) 
landscape = np.zeros((len(gammas), len(betas)))

for i, g in enumerate(gammas):
    for j, b in enumerate(betas):
        landscape[i, j] = cost_function(np.array([[g], [b]]))

plt.imshow(landscape, origin="lower",
           extent=[0, np.pi, 0, 2*np.pi], cmap="RdYlBu_r")

plt.colorbar(label=r"$\langle H_C \rangle$")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$\gamma$")
plt.savefig("qaoa_landscape.png", dpi=300)

# Find global minimum
min_idx    = np.unravel_index(np.argmin(landscape), landscape.shape)
best_gamma = gammas[min_idx[0]]
best_beta  = betas[min_idx[1]]
print(f"Global minimum at γ={best_gamma:.4f}, β={best_beta:.4f}, "
      f"⟨H_C⟩={landscape[min_idx]:.4f}")