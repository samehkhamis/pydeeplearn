# Recursive Neural Network (RNN)
# Objective function is varying w.r.t. the input sentence / dependency tree
# Author: Sameh Khamis (sameh@umiacs.umd.edu)
# License: GPLv2 for non-commercial research purposes only

import numpy as np
from ..core.net import *

class RNN(Net):
    def __init__(self, wordvecs, update=RMSprop(), step=Inverse(), name='net', root_dir=None):
        self.setup(update, step, name, root_dir)
        
        # The wordvecs object is used to initialize word representations to glove/word2vec
        d = wordvecs._d
        self._wordvecs = wordvecs
        self._w = Param(np.r_[np.eye(d), np.eye(d)] * 0.5 + np.random.randn(2 * d, d) * 0.01)
        self._b = Param.zeros((d,))
        self._wscore = Param.randn((d, 5))
        self._bscore = Param.zeros((5,))
        self._words = {}
    
    def _get_param(self, word):
        word_lower = word.lower()
        if word_lower not in self._words:
            wordvec = self._wordvecs.get_vector(word_lower).reshape(1, -1)
            self._words[word_lower] = Param(wordvec)
        return self._words[word_lower]
    
    def parse(self, trees):
        if not isinstance(trees, list):
            trees = [trees]
        
        lambdaa = 1e-4
        loss_weight = 1.0 / np.sum([tree.size for tree in trees])
        self._obj = lambdaa * (Sum(self._w**2) + Sum(self._wscore**2))
        for tree in trees:
            self._obj = self._obj + loss_weight * self._parse(tree)
        
        self.process_objective()
    
    def _parse(self, tree):
        s1 = [tree]    # pre-order stack
        s2 = []        # post-order stack
        
        while len(s1) > 0:
            node = s1.pop()
            if node.left is not None: # and node.right is not None
                s1.append(node.left)
                s1.append(node.right)
            s2.append(node)
        
        expr = {}
        obj = None
        
        while len(s2) > 0:
            node = s2.pop()
            if node.left is not None: # and node.right is not None
                expr[node] = Tanh(Affine(Concat(expr[node.left], expr[node.right]), self._w, self._b))
            else:
                expr[node] = self._get_param(node.value)
            
            label = Label()
            label._value = np.array([int(node.label)])
            if obj is None:
                obj = Softmax(Affine(expr[node], self._wscore, self._bscore), label)
            else:
                obj = obj + Softmax(Affine(expr[node], self._wscore, self._bscore), label)
        
        return obj
    
    def predict(self, trees):
        labels = []
        gt = []
        
        for i in np.arange(len(trees)):
            # set the data and the labels
            self.parse(trees[i])
            
            # calculate result and add to list
            self.forward()
            labels.append(self.result)
            gt.append(self.groundtruth)
        
        return (np.concatenate(labels), np.concatenate(gt))
    
    def train(self, trees, batchsize=100, epochs=10, progress=10, snapshot=-1):
        if batchsize > len(trees): batchsize = len(trees)
        
        for e in np.arange(epochs):
            idx = self.get_random_indices(len(trees), batchsize)
            
            for i in np.arange(0, idx.size, batchsize):         # for each batch in this epoch
                # set the input to the current batch
                self.parse(trees[i: i + batchsize])
                
                # calculate f(x) and f'(x)
                self.forward()
                self.backward()
                
                # calculate the updates and apply then
                alpha = self._step.get(self._iter)
                self._update.apply([p for p in self._params if not p._fixed], alpha)
                
                if progress > 0 and (self._iter + 1) % progress == 0:
                    self.print_progress()
                if snapshot > 0 and (self._iter + 1) % snapshot == 0:
                    filename = os.path.join(self._cwd, '%s_%012d.pkl' % (self._name, self._iter + 1))
                    Net.save(self, filename)
                
                self._iter += 1
            self._epoch += 1
