import time 
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# gen random graph
seed = 10608068 
np.random.seed(seed)
G = nx.gnp_random_graph(n=8, p=0.5, seed=seed)
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
print(f"Edges: {list(G.edges())}")

# Plot the graph
pos = nx.spring_layout(G)
nx.draw(G, pos=pos, with_labels=True, node_color="skyblue")
plt.savefig("random_graph.png", dpi=300)

# Brute-force search for maximum cut
n_nodes = G.number_of_nodes()
max_cut = 0
optimal_partitions = []

t0 = time.time()
for case_i in range(2 ** n_nodes):
    bitstring = np.binary_repr(case_i, n_nodes)   # e.g. "01101010"
    partition = {n: int(bitstring[n]) for n in range(n_nodes)}
    cut_value = sum(
        1 for u, v in G.edges() if partition[u] != partition[v]
    )
    if cut_value > max_cut:
        max_cut = cut_value
        optimal_partitions = [bitstring]
    elif cut_value == max_cut:
        optimal_partitions.append(bitstring)
elapsed = time.time() - t0
print(f"Brute-force: max cut={max_cut}, time={elapsed}s")

print(f"\nMaximum cut value: {max_cut}")
print(f"Optimal partitions ({len(optimal_partitions)} found):")
for p in optimal_partitions:
    print(f"  {p}  →  S0={[i for i,b in enumerate(p) if b=='0']}, "
          f"S1={[i for i,b in enumerate(p) if b=='1']}")

# Visualize one optimal partition solution
best = optimal_partitions[0]
partition = {n: int(best[n]) for n in range(n_nodes)}

node_colors = ["skyblue" if partition[n] == 0 else "salmon" for n in G.nodes()]
cut_edges    = [(u, v) for u, v in G.edges() if partition[u] != partition[v]]
non_cut_edges = [(u, v) for u, v in G.edges() if partition[u] == partition[v]]

plt.clf()
plt.cla()
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, edgelist=non_cut_edges,
                       edge_color="gray", width=1.5)
nx.draw_networkx_edges(G, pos, edgelist=cut_edges,
                       edge_color="red", width=2.5, style="dashed")

plt.savefig("max_cut_solution.png", dpi=300)
