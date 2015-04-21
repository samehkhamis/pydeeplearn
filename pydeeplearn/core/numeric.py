# Numerically stable routines, and some other helper functions
# Guarded against loss of significance and/or catastrophic cancellation
# Author: Sameh Khamis (sameh@umiacs.umd.edu)
# License: GPLv2 for non-commercial research purposes only

import numpy as np

def eps(dtype=np.float32):
    return np.finfo(dtype).eps

def root(x, n): # the positive real nth root of x
    r = np.roots([1] + [0]*(n - 1) + [-x])
    r = np.real(r)[(np.imag(r) == 0) & (r > 0)]
    return r[0]

def sigmoid(x): # 1 / (1 + exp(-x))
    mask = x >= 0
    result = np.empty(x.shape, dtype=x.dtype)
    result[mask] = 1 / (1 + np.exp(-x[mask]))
    z = np.exp(x[~mask])
    result[~mask] = z / (1 + z)
    return result

def log1p(x): # log(1 + x)
    mask = np.abs(x) < 0.0070981273157505465 # root(3 * eps(), 3), 2nd order Taylor approx.
    result = np.empty(x.shape, dtype=x.dtype)
    xmask = x[mask]
    result[mask] = xmask - xmask*xmask / 2 # log(1 + x) = x - x^2 / 2 for small x
    result[~mask] = np.log(1 + x[~mask])
    return result

def expm1(x): # exp(x) - 1
    mask = np.abs(x) < 0.0070981273157505465 # root(3 * eps(), 3), 2nd order Taylor approx.
    result = np.empty(x.shape, dtype=x.dtype)
    xmask = x[mask]
    result[mask] = xmask + xmask*xmask / 2 # exp(x) - 1 = x + x^2 / 2 for small x
    result[~mask] = np.exp(x[~mask]) - 1
    return result

def log1pexp(x): # log(1 + exp(x))
    mask = x >= 0
    result = np.empty(x.shape, dtype=x.dtype)
    result[mask] = x[mask] + log1p(np.exp(-x[mask]))
    result[~mask] = log1p(np.exp(x[~mask]))
    return result

def logsumexp(x): # log(sum(exp(x_i)))
    a = x.max(axis=1).reshape(-1, 1)
    return np.log(np.sum(np.exp(x - a), axis=1)) + a

def softmax(x): # exp(x_c) / sum(exp(x_i))
    result = np.exp(x - x.max(axis=1).reshape(-1, 1))
    z = result.sum(axis=1).reshape(-1, 1)
    return result / z

def onehot(labels, C):
    result = np.zeros((labels.size, C), dtype=labels.dtype)
    result[np.arange(labels.size), labels.reshape(-1)] = 1
    return result
