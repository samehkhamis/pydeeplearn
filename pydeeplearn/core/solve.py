# Stochastic gradient descent support
# Updates included are vanilla, momentum, Nesterov's, Adagrad, and RMSprop
# Also supports fixed and inverse step decays
# Author: Sameh Khamis (sameh@umiacs.umd.edu)
# License: GPLv2 for non-commercial research purposes only

import numpy as np

class ParamDict(dict):
    def __missing__(self, param):
        self[param] = np.zeros(param.value.shape, dtype=param.dtype)
        return self[param]

class Step:
    def __init__(self, alpha0):
        self._alpha0 = alpha0
    
    def get(self):
        return self._alpha0

class FixedDecay(Step):
    def __init__(self, alpha0=0.01, niter=100, gamma=0.95):
        self._alpha0 = alpha0
        self._niter = niter
        self._gamma = gamma
    
    def get(self, t):
        return self._alpha0 * self._gamma**int(t / self._niter)

class InverseDecay(Step):
    def __init__(self, alpha0=0.01, gamma=0.01, degree=1.0):
        self._alpha0 = alpha0
        self._gamma = gamma
        self._degree = degree
    
    def get(self, t):
        return self._alpha0 / (1 + self._gamma * t)**self._degree

class ExponentialDecay(Step):
    def __init__(self, alpha0=0.01, gamma=0.0005):
        self._alpha0 = alpha0
        self._gamma = gamma
    
    def get(self, t):
        return self._alpha0 * np.exp(-self._gamma * t)

class Update: # Vanilla SGD (no momentum)
    def apply(self, params, rate):
        for param in params:
            param._value += -rate * param.gradient
    
    def __setstate__(self, dict):
        self.__dict__ = dict
    
    def __getstate__(self):
        dict = self.__dict__.copy()
        dict['_params'] = []
        if '_updates' in dict: dict['_updates'] = []
        if '_cache' in dict: dict['_cache'] = []
        return dict

class NAG(Update):
    def __init__(self, momentum=0.9):
        self._momentum = momentum
        self._ahead_momentum = momentum
        self._updates = ParamDict()
    
    def apply(self, params, rate):
        for param in params:
            old_update = self._updates[param]
            self._updates[param] = self._momentum * old_update - rate * self._param.gradient
            param._value += -self._ahead_momentum * old_update + (1 + self._ahead_momentum) * self._updates[param]

class SGD(NAG):
    def __init__(self, momentum=0.9):
        self._momentum = momentum
        self._ahead_momentum = 0
        self._updates = ParamDict()

class RMSprop(Update):
    def __init__(self, cache_decay=0.99):
        self._cache_weight = cache_decay
        self._gradient_weight = 1 - cache_decay
        self._cache = ParamDict()
    
    def apply(self, params, rate):
        for param in params:
            self._cache[param] = self._cache_weight * self._cache[param] + self._gradient_weight * param.gradient**2
            param._value += -rate * param.gradient / (np.sqrt(self._cache[param]) + 1e-6)

class Adagrad(RMSprop):
    def __init__(self):
        self._cache_weight = 1
        self._gradient_weight = 1
        self._cache = ParamDict()
