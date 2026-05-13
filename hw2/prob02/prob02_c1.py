import time
import networkx as nx
import numpy as np
import pennylane as qp
from pennylane import qaoa
from pennylane import numpy as pnp

# gen random graph
seed = 10608068 
np.random.seed(seed)
G = nx.gnp_random_graph(n=8, p=0.5, seed=seed)
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
print(f"Edges: {list(G.edges())}")

# QAOA
n_nodes = G.number_of_nodes()
wires = range(n_nodes)
cost_h, mixer_h = qaoa.maxcut(G)
max_cut = 11

def qaoa_layer(gamma, alpha):
    qaoa.cost_layer(gamma, cost_h)
    qaoa.mixer_layer(alpha, mixer_h)

def circuit(params, depth,  **kwargs):
    for w in wires:
        qp.Hadamard(wires=w)
    qp.layer(qaoa_layer, depth, params[0], params[1]) 

dev = qp.device("default.qubit", wires=wires)
@qp.qnode(dev)
def cost_function(params, depth=1):
    circuit(params, depth=depth)
    return qp.expval(cost_h)

def optimize_qaoa(depth):
    optimizer = qp.GradientDescentOptimizer()
    steps = 300 * depth
    params = pnp.array(
        [[0.5] * depth, [0.5] * depth], requires_grad=True)

    def _cost_fn(params):
        return cost_function(params, depth=depth)

    for _ in range(steps):
        params = optimizer.step(_cost_fn, params)
    return params

@qp.qnode(dev)
def probability_circuit(gamma, alpha, depth=1):
    circuit([gamma, alpha], depth=depth)
    return qp.probs(wires=wires)

for depth in [1, 2, 3, 4]:
    t0 = time.time()
    params = optimize_qaoa(depth)
    elapsed = time.time() - t0    

    probs       = probability_circuit(params[0], params[1], depth=depth)
    best_idx    = int(np.argmax(probs))
    best_bits   = np.binary_repr(best_idx, n_nodes)
    best_cut    = sum(1 for u, v in G.edges()
                      if int(best_bits[u]) != int(best_bits[v]))
    approx_ratio = best_cut / max_cut

    print(f"p={depth}: best bits: {best_bits}, best cut={best_cut}/{max_cut}, "
          f"approx ratio={approx_ratio:.4f}, "
          f"γ={params[0].numpy()}, β={params[1].numpy()}, "
          f"time={elapsed:.2f}s")
