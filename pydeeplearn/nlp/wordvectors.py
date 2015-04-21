# Word vectors class, supports projection and analogies
# Reads GloVe format (Pennington et al), easy to extend to word2vec (Mikolov et al)
# Author: Sameh Khamis (sameh@umiacs.umd.edu)
# License: GPLv2 for non-commercial research purposes only

import numpy as np
import gzip, urllib, os

class WordVectors:
    def __init__(self, d, n):
        self._d = d
        self._n = n
        
        self._worddict = {}
        self._reversedict = {}
        self._wordmatrix = np.random.randn(n, d).astype(np.float32) * 0.001
        self._wordnorms = np.linalg.norm(self._wordmatrix, axis=1)
        self._unknown = np.mean(self._wordmatrix, axis=0)
    
    @staticmethod
    def from_glove(d=300):
        vectorsfile = 'glove.6B.%dd.txt.gz' % d
        if not os.path.exists(vectorsfile):
            urllib.urlretrieve('http://www-nlp.stanford.edu/data/' + vectorsfile, vectorsfile)
        
        wordvectors = WordVectors(d, 400000)
        i = 0
        with gzip.open(vectorsfile, 'r') as f:
            for line in f:
                idx = line.find(' ')
                wordvectors._worddict[line[:idx]] = i
                wordvectors._reversedict[i] = line[:idx]
                wordvectors._wordmatrix[i] = np.array(line[idx + 1:].split(' '), dtype=np.float32)
                i += 1
        
        wordvectors._wordnorms = np.linalg.norm(wordvectors._wordmatrix, axis=1)
        wordvectors._unknown = np.mean(wordvectors._wordmatrix, axis=0)
        return wordvectors
    
    @staticmethod
    def from_word2vec(d):
        raise Exception("not implemented yet!")
    
    def project(self, W):
        assert W.shape[0] == self._d
        self._d = W.shape[1]
        self._wordmatrix = self._wordmatrix.dot(W.astype(np.float32))
        self._wordnorms = np.linalg.norm(self._wordmatrix, axis=1)
        self._unknown = np.mean(self._wordmatrix, axis=0)
    
    def get_vector(self, word):
        if word not in self._worddict: return self._unknown
        idx = self._worddict[word]
        return self._wordmatrix[idx]
    
    def get_analogy(self, worda, wordb, wordc, k=10):
        veca = self.get_vector(worda)
        vecb = self.get_vector(wordb)
        vecc = self.get_vector(wordc)
        vecd = vecb - veca + vecc
        cossim = np.dot(self._wordmatrix, vecd) / self._wordnorms / np.linalg.norm(vecd)
        top10 = np.argsort(cossim)[-k:][::-1]
        return [(self._reversedict[i], cossim[i]) for i in top10]
