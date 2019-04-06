import numpy as np
from scipy import stats

def phi(x):
    g=np.sqrt(2)/2*(3/4*stats.multivariate_normal.pdf(x,[0,np.pi,0,0,np.pi,0,0,0,0,0]) + 1/4*stats.multivariate_normal.pdf(x,[0,np.pi,0,0,-np.pi,0,0,0,0,0]))
    return g
