# Refer: https://en.wikipedia.org/wiki/Test_functions_for_optimization

import numpy as np
import matplotlib.pyplot as plt
import pso_solver

pso_solver.setSeed(1)

lower = xlower = ylower = -5
upper = xupper = yupper = 5
particleList = pso_solver.createRandomParticleList(2, numParticles=10, lower=lower, upper=upper)

# Testing on Rastrigin function with n = 2
f = lambda x, y: 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))

pso_solver.psoVisualizeBivariate(particleList, f, xlower, xupper, ylower, yupper, c1=1, c2=1, W=0.5, numIters=50, maxFlag=False, sleepTime=0.1, density=100, accuracy=2, verbose=False)
