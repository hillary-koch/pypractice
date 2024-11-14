import numpy as np

def logsumexp(lp):
    y = np.max(lp)
    return np.log(np.sum(np.exp(lp-y), keepdims=True)) + y