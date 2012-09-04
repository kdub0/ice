import indexing
from numpy import *

def create(N):
    assert N > 1
    actions = [4]*N
    idx     = indexing.create(actions)
    (M,_,_) = idx
    U       = [zeros((M,4)) for i in range(N)]
    for outcome in range(M):
        a    = indexing.unindex(idx, outcome)
        left = 0
        back = 0
        for x in a:
            if x%2 == 0:
                left += 1
            if x/2 == 0:
                back += 1

        v = array([1.5+.2*N, 9.0, 1.0/8.0+0.04*N, 7+0.4*N])
        for i in range(N):
            if a[i]%2 == 0:
                u = v + array([1.0, 1.5, 1.0/20.0, 2.0])
            else:
                u = v + array([1.0+2.0*left, 1.0, 1.0/20.0+0.04*left, 1.5+.4*left])

            if a[i]/2 == 0:
                u = v + array([6.0+0.5*back, 12.0, 1.0/7.0+0.01*back, 10.0+3.0*back])
            else:
                u = v + array([2.0, 20.0, 1.0/8.0, 15.0])
            
            U[i][outcome,:] = -u

    return (idx,U)
