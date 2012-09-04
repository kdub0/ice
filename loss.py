import math

"""
compute logloss of smoothed prediction
"""
def logloss(truth, prediction, epsilon=0.0):
    N = len(truth)
    assert len(prediction) == N

    loss = 0.0
    for i in range(N):
        if truth[i] > 1e-15:
            loss -= truth[i]*math.log((1-epsilon)*prediction[i] + epsilon/N)

    return loss/math.log(2)
