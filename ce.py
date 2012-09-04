import indexing
from numpy import *

"""
game -> maxent ce
"""
def solve(game, c, w, opt):
    (index,U) = game
    (M,actions,_) = index
    N = len(actions)
    K = U[0].shape[1]

    deviation_offset = [0]*N
    offset           = 0
    for i in range(N):
        deviation_offset[i] = offset
        offset              = offset + actions[i]*(actions[i]-1)
    deviations = offset

    Uw = zeros((M, N))
    for outcome in range(M):
        for i in range(N):
            Uw[outcome,i] = dot(U[i][outcome], w)

    cp = zeros(M)
    for outcome in range(M):
        for i in range(N):
            cp[outcome] += c[i]*Uw[outcome,i]

    R = zeros((M, deviations))
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
                    R[outcome,deviation] = Uw[outcomep,i] - Uw[outcome,i]

    def predict(x):
        regrets = cp - dot(R,x)
        offset  = regrets.max()
        sigma   = exp(regrets-offset)
        Z       = sum(sigma)
        return sigma/Z

    def gradient(x):
        sigma = predict(x)
        g     = dot(sigma, R)
        return g

    return predict(opt(deviations, gradient))
