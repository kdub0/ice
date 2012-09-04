import math
from numpy import *

def Rprox(w, gamma, g):
    return w - gamma*g

def Rpprox(w, gamma, g):
    return maximum(w - gamma*g, 0)

"""
stochastic gradient descent
"""
def create(T, L, l1, l2, prox=Rprox):
    def optimizer(K, gradient):
        w = zeros(K)
        x = zeros(K)
        for t in range(T):
            g  = gradient(w) + copysign(l1, w) + l2*w
            w  = prox(w, 1.0/math.sqrt(T)/L, g)
            x += w/(T+1)
        return x

    return optimizer
