import numpy as np
import matplotlib.pyplot as plt
import pso_solver

pso_solver.setSeed()

lower = xlower = 0
upper = xupper = 2 * np.pi
particleList = pso_solver.createRandomParticleList(1, numParticles=10, lower=lower, upper=upper)

# Testing on a univariate function
f = lambda x: np.cos(x)

pso_solver.psoVisualizeUnivariate(particleList, f, xlower, xupper, c1=1, c2=1, W=0.5, numIters=30, maxFlag=False, sleepTime=0.1, density=100, fColor='b', particleColor='r', accuracy=2, verbose=False)
