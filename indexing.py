"""
create a game indexing
"""
def create(actions):
    N = len(actions)
    M = 1
    
    offsets = [0]*N
    for i in range(N-1,-1,-1):
        assert actions[i] > 0

        offsets[i] = M
        M          = M*actions[i]

    return (M, actions, offsets)

"""
actions -> outcome
"""
def index(indexing, action):
    (_,actions,_) = indexing
    
    outcome = 0
    for a,n in zip(action,actions):
        assert a >= 0 and a < n
        outcome = n*outcome + a

    return outcome

"""
outcome -> actions
"""
def unindex(indexing, outcome):
    (M,actions,_) = indexing
    N = len(actions)
    assert outcome >= 0 and outcome < M
    
    action = [0]*N
    for i in range(N-1,-1,-1):
        action[i] = outcome%actions[i]
        outcome   = outcome/actions[i]

    return action

"""
outcome -> player i's action 
"""
def unindexi(indexing, outcome, i):
    (M,actions,offsets) = indexing
    N = len(actions)
    assert outcome >= 0 and outcome < M
    assert i >= 0 and i < N
    
    return outcome/offsets[i]%actions[i]

"""
outcome -> outcome, switching i's action to y
"""
def reindex(indexing, outcome, i, y):
    (M,actions,offsets) = indexing
    N = len(actions)
    assert outcome >= 0 and outcome < M
    assert i >= 0 and i < N
    assert y >= 0 and y < actions[i]

    high = outcome/offsets[i]/actions[i]
    low  = outcome%offsets[i]
    
    return (high*actions[i] + y)*offsets[i] + low
