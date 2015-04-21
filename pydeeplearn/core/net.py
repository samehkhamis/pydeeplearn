# Base class for a neural network (or any function composition)
# Works with functions connected in a general DAG (not just a chain)
# Also supports snapshot saving and loading, and exporting to Graphviz dot files
# Author: Sameh Khamis (sameh@umiacs.umd.edu)
# License: GPLv2 for non-commercial research purposes only

import numpy as np
from datetime import datetime
import os, cPickle
from solve import *
from layers import *

class Net:
    def __init__(self, objective, update=RMSprop(), step=Inverse(), name='net', root_dir=None):
        self.setup(update, step, name, root_dir)
        
        self._obj = objective
        self.process_objective()
    
    def setup(self, update=RMSprop(), step=Inverse(), name='net', root_dir=None):
        self._update = update
        self._step = step
        self._epoch = 0
        self._iter = 0
        self._cwd = root_dir if root_dir is not None else os.getcwd()
        self._name = name
    
    def process_objective(self):
        self._params = set()
        self._labels = set()
        self._data = set()
        
        self.toposort()
        
        self._params = list(self._params)
        self._data = list(self._data)
        self._labels = list(self._labels)
    
    @property
    def result(self):
        return np.concatenate([x.result for x in self._labels])
    
    @property
    def groundtruth(self):
        return np.concatenate([x.value for x in self._labels])
    
    @property
    def value(self):
        return self._obj.value
    
    def toposort(self):
        # Recursive depth-first DAG topological sort (from one sink node)
        visited = set()
        self._sorted_nodes = []
        self._toposort(self._obj, visited)
    
    def _toposort(self, node, visited):
            visited.add(node)
            
            # Also, collect param, data, and label nodes
            if isinstance(node, Param):
                self._params.add(node)
            elif isinstance(node, Data):
                self._data.add(node)
            elif isinstance(node, Label):
                self._labels.add(node)
            
            for x in node.input:
                if not x in visited:
                    self._toposort(x, visited)
            self._sorted_nodes.append(node)
    
    def forward(self, dropout=True):
        for node in self._sorted_nodes:
            if isinstance(node, Dropout):
                node.forward(disabled=not dropout)
            else:
                node.forward()
    
    def backward(self, dropout=True):
        self._obj._gradient[:] = 1
        
        for node in reversed(self._sorted_nodes):
            if isinstance(node, Dropout):
                node.backward(disabled=not dropout)
            else:
                node.backward()
    
    def print_progress(self):
        gnorm = np.linalg.norm(np.concatenate([p.gradient.flatten() for p in self._params]))
        ts = datetime.now().strftime('%b-%d %I:%M:%S %p')
        print '%s    epoch: %d, iter: %d, cost: %.3f, gnorm: %.3f' % (ts, self._epoch + 1, self._iter + 1, self.value[0], gnorm)
    
    def to_dot(self, filename):
        node_index = {}
        for i in np.arange(len(self._sorted_nodes)):
            node_index[self._sorted_nodes[i]] = i
        
        f = open(filename, 'w')
        f.write('digraph graphname {\n')
        for node in self._sorted_nodes:
            f.write('%d [label="%s"];\n' % (node_index[node], str(node)))
            if isinstance(node, Op):
                f.write('%d [shape=box];\n' % (node_index[node]));
        for node2 in self._sorted_nodes:
            for node1 in node2.input:
                f.write('%d -> %d;\n' % (node_index[node1], node_index[node2]))
        f.write('}\n')
        f.close()
    
    @staticmethod
    def save(net, filename):
        f = open(filename, 'wb')
        net._iter += 1
        cPickle.dump(net, f)
        net._iter -= 1
        f.close()
    
    @staticmethod
    def load(filename):
        f = open(filename, 'rb')
        net = cPickle.load(f)
        net._cwd = os.path.dirname(filename)
        f.close()
        return net
    
    def get_random_indices(self, n, batchsize):
        extra = (-(n % batchsize)) % batchsize
        idx = np.random.permutation(n + extra)  # randomly permute the samples
        idx %= n                                # exactly divides batchsize
        return idx
    
    def gradient_check(self, h=1e-5):
        numerical = [np.empty(param.gradient.shape, dtype=param.dtype) for param in self._params]
        analytic = []
        
        for i in np.arange(len(self._params)):
            param = self._params[i]
            for d in np.arange(param.value.size):
                param._value.flat[d] += h               # set to f(x + h)
                f1 = self.forward(dropout=False)
                param._value.flat[d] -= 2 * h           # set to f(x - h)
                f2 = self.forward(dropout=False)
                param._value.flat[d] += h               # reset
                numerical[i].flat[d] = (f1 - f2) / (2 * h)
            self.forward(dropout=False)
            self.backward(dropout=True)
            analytic.append(param.gradient)             # analytic gradient f'(x)
        
        numerical = np.concatenate([x.flatten() for x in numerical])
        analytic = np.concatenate([x.flatten() for x in analytic])
        return np.abs(numerical - analytic) / (np.abs(numerical) + np.abs(analytic))
