import indexing
import ice
from numpy import *
import scipy.sparse

"""
game -> instance
"""
def to_instance(game):
    ((index,U),demonstrated) = game
    
    (M,actions,_) = index
    N = len(actions)
    K = U[0].shape[1]

    deviation_offset = [0]*N
    offset           = 0
    for i in range(N):
        deviation_offset[i] = offset
        offset              = offset + actions[i]*(actions[i]-1)
    deviations = offset

    A   = []
    row = []
    col = []
    for outcome in range(M):
        for i in range(N):
            action = indexing.unindex(index, outcome)
            x      = action[i]
            for y in range(actions[i]):
                if x != y:
                    outcomep = indexing.reindex(index, outcome, i, y)

                    deviation = deviation_offset[i] + (actions[i]-1)*x + y
                    if x < y:
                        deviation = deviation - 1
                    for k in range(K):
                        A.append(U[i][outcomep,k] - U[i][outcome,k])
                        row.append(outcome)
                        col.append(K*deviation+k)                   
        
    return (demonstrated, K, scipy.sparse.csr_matrix((A,(row,col)), shape=(M, K*deviations)))

"""
instance -> dual weights
"""
def solve(inst, opt):
    (demonstrated, K, R) = inst
    W                    = R.shape[1]
    deviations           = W/K

    demonstrations = demonstrated*R

    row      = zeros(deviations*K, dtype=int)
    col_base = zeros(deviations*K, dtype=int)
    for alt in range(deviations):
        for k in range(K):
            pos           = K*alt + k
            row[pos]      = alt
            col_base[pos] = k

    alpha = [1.0]*deviations
    A     = []
    for k in range(deviations):
        A.append(scipy.sparse.csr_matrix((demonstrations, (row, col_base+K*k)), shape=(deviations, W)))

    beta = [1.0]
    B    = [R]

    return ice.solve(zeros(W), alpha, A, beta, B, opt)
    
"""
instance, dual weights -> prediction
"""
def predict(inst, w):
    (_, _, R) = inst
    return ice.predict(R, w)
