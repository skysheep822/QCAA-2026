import dimod
import neal
import time
import networkx as nx
import numpy as np

seed = 10608068 
np.random.seed(seed)
G = nx.gnp_random_graph(n=8, p=0.5, seed=seed)

n_nodes = G.number_of_nodes()
wires = range(n_nodes)
max_cut = 11

def maxcut_to_qubo(G):
    Q = {}
    for u, v in G.edges():
        Q[(u, u)] = Q.get((u, u), 0) - 1
        Q[(v, v)] = Q.get((v, v), 0) - 1
        Q[(u, v)] = Q.get((u, v), 0) + 2
    return Q

Q_mc  = maxcut_to_qubo(G)
bqm_mc = dimod.BQM.from_qubo(Q_mc)

t0 = time.time()
sa_sampler = neal.SimulatedAnnealingSampler()
sa_result  = sa_sampler.sample(bqm_mc, num_reads=1000, seed=seed)
sa_elapsed = time.time() - t0

sa_best     = sa_result.first
sa_bits     = "".join(str(sa_best.sample[i]) for i in range(n_nodes))
sa_cut      = sum(1 for u, v in G.edges() if int(sa_bits[u]) != int(sa_bits[v]))
sa_approx   = sa_cut / max_cut
print(f"SA: best bits={sa_bits}, best cut={sa_cut}, approx ratio={sa_approx:.4f}, time={sa_elapsed:.2f}s")