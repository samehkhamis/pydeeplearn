# Convolutional Neural Network (CNN)
# Objective function (or network structure) is fixed and given as input
# Author: Sameh Khamis (sameh@umiacs.umd.edu)
# License: GPLv2 for non-commercial research purposes only

import numpy as np
from ..core.net import *

# TODO: multiple data / labels, data augmentations, deconv layer
class CNN(Net):
    def set_input(self, data, labels=None):
        # CNNs assumed to have one data node and one label node for now
        self._data[0]._value = data if data.ndim == 4 else data[np.newaxis]
        self._labels[0]._value = labels if labels is not None else np.zeros(1, dtype=np.int32)
    
    def predict(self, data, batchsize=100):
        labels = []
        if data.ndim < 4: data = data[np.newaxis]
        
        for i in np.arange(0, data.shape[0], batchsize):
            # set the data and the labels
            self.set_input(data[i:i + batchsize])
            
            # calculate result and add to list
            self.forward(dropout=False)
            labels.append(self.result)
        return np.concatenate(labels)
    
    def train(self, data, labels, batchsize=100, epochs=10, progress=10, snapshot=-1):
        if batchsize > data.shape[0]: batchsize = data.shape[0]
        
        for e in np.arange(epochs):
            idx = self.get_random_indices(data.shape[0], batchsize)
            
            for i in np.arange(0, idx.size, batchsize):         # for each batch in this epoch
                # set the input to the current batch
                batchidx = idx[i:i + batchsize]
                self.set_input(data[batchidx], labels[batchidx])
                
                # calculate f(x) and f'(x)
                self.forward(dropout=True)
                self.backward(dropout=True)
                
                # calculate the updates and apply then
                alpha = self._step.get(self._iter)
                self._update.apply(self._params, alpha)
                
                if progress > 0 and (self._iter + 1) % progress == 0:
                    self.print_progress()
                if snapshot > 0 and (self._iter + 1) % snapshot == 0:
                    filename = os.path.join(self._cwd, '%s_%012d.pkl' % (self._name, self._iter + 1))
                    Net.save(self, filename)
                
                self._iter += 1
            self._epoch += 1
