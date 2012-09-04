from numpy import *
import random

"""
log-linear prediction
"""
def predict(b, x):
    regrets = -b*x
    offset  = max(regrets)
    sigma   = exp(regrets-offset)
    Z       = sum(sigma)
    return sigma/Z

"""
solve optimizations of the form:

min_x   c'*x + sum_i alpha_i*max_k (A^(i)_k*x)^+ + sum_j beta_j*log sum_k exp(B^(j)_k*x)
"""
def solve(c, alpha, A, beta, B, opt):
    N = len(alpha)
    assert len(A) == N

    M = len(beta)
    assert len(B) == M

    K = A[0].shape[1]
    for a in A:
        assert a.shape[1] == K
    
    for b in B:
        assert b.shape[1] == K

    alphaZ = sum(alpha)
    betaZ  = sum(beta)

    def sample(pi, Z):
        dart = random.random()*Z

        for i, p in enumerate(pi):
            if dart < p:
                return i
            dart -= p

        return 0

    def gradient(x):
        g  = zeros_like(x)
        g += c

        i = sample(alpha, alphaZ)
        a = A[i]
        u = alpha[i]*alphaZ

        phi = a*x
        alt = argmax(phi)
        if phi[alt] > 0:
            g += u*a[alt].toarray().flatten()

        j = sample(beta, betaZ)        
        b = B[j]
        v = beta[j]*betaZ

        sigma = predict(b,x)
        g     = g - v*(sigma*b)

        return g

    return opt(K, gradient)



