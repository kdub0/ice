from numpy import *
import random

"""
draw M samples from sigma, return demonstrated behaviour
"""
def draw(M, sigma):
    sigma = list(sigma)
    N     = len(sigma)
    Z     = 0
    for i in range(N):
        Z       += sigma[i]
        sigma[i] = Z
    darts = [random.random()*Z for i in range(M)]
    darts.sort()
    
    demon = zeros(N)
    i = 0
    for j in range(N):
        while i < M and darts[i] < sigma[j]:
            demon[j] += 1.0/M
            i        += 1

    return demon
