# Refer: https://arxiv.org/pdf/1308.4008.pdf (Momin, J. A. M. I. L., & Yang, X. S. (2013). A literature survey of benchmark functions for global optimization problems. Journal of Mathematical Modelling and Numerical Optimisation, 4(2), 150-194.)

import numpy as np
import matplotlib.pyplot as plt
import pso_solver

pso_solver.setSeed(1)
lower = xlower = ylower = zlower = -20
upper = xupper = yupper = zupper = 20
particleList = pso_solver.createRandomParticleList(3, numParticles=100, lower=lower, upper=upper)

# Testing on Schwefel 2.4 function
f = lambda x, y, z: ((x-1)**2+(y-1)**2+(z-1)**2) + (x-x**2)**2+(x-y**2)**2+(x-z**2)**2

pso_solver.psoVisualizeTrivariate(particleList, f, xlower, xupper, ylower, yupper, zlower, zupper, c1=1, c2=1, W=0.5, numIters=20, maxFlag=False, sleepTime=0.5, accuracy=2, verbose=False)
