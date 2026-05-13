import numpy as np
np.set_printoptions(suppress=True, precision=0, floatmode='fixed')

def build_qubo(weights, values, max_weight, lambd):
    M = int(np.ceil(np.log2(max_weight))) 
    N = len(weights) + M
    w_ext = np.array(weights + [2**k for k in range(M)])
    v_ext = np.array(values + [0] * M)
    
    Q = np.zeros((N, N)) # initialize zero
    for i in range(N):
        Q[i, i] = -v_ext[i] + lambd * w_ext[i] * (w_ext[i] - 2 * max_weight)
    for i in range(N):
        for j in range(i + 1, N):
            Q[i, j] = 2 * lambd * w_ext[i] * w_ext[j]
    return Q

if __name__ == "__main__":
    weights = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
    values  = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
    max_weight = 165

    Q = build_qubo(weights, values, max_weight, lambd=2)
    print(Q.astype(int))