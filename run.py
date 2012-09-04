import ce
import game
import loss
import numpy
import routing
import sample
import sgd

N = 4      # number of players
T = 100000 # SGD iterations
M = 100    # samples of play
eps = 0.01 # multinomial smoothing
C   = 0.01 # CE max welfare coefficient
c   = C*numpy.ones(N);
w   = numpy.array([0.0,0.0,0.0,1.0]) # true utility function


g     = routing.create(N)
truth = ce.solve(g, c, w, sgd.create(T, 1, 0, 0, sgd.Rpprox))
demon = sample.draw(M, truth) 

inst  = game.to_instance((g,demon))
w     = game.solve(inst, sgd.create(T, 1, 0, 0))
pred  = game.predict(inst, w)

print 'multinomial loss', loss.logloss(truth, demon, eps)
print 'ice loss', loss.logloss(truth, pred)
print 'truth entropy', loss.logloss(truth, truth)
