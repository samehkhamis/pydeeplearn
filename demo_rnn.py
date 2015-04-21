# Sentiment Analysis: RNN on Movie Reviews
# Author: Sameh Khamis (sameh@umiacs.umd.edu)
# License: GPLv2 for non-commercial research purposes only

import numpy as np
import urllib, os
from zipfile import ZipFile
from pydeeplearn.core.solve import RMSprop, Inverse
from pydeeplearn.nlp.ptbtree import PTBTree
from pydeeplearn.nlp.wordvectors import WordVectors
from pydeeplearn.net.rnn import RNN

# Load the dataset files
ptbfile = 'trainDevTestTrees_PTB.zip'
if not os.path.exists(ptbfile):
    urllib.urlretrieve('http://nlp.stanford.edu/sentiment/' + ptbfile, ptbfile)

f = ZipFile(ptbfile)
train_trees = [PTBTree.parse(line) for line in f.read('trees/train.txt').split('\n') if len(line) > 0]
dev_trees = [PTBTree.parse(line) for line in f.read('trees/dev.txt').split('\n') if len(line) > 0]
test_trees = [PTBTree.parse(line) for line in f.read('trees/test.txt').split('\n') if len(line) > 0]
f.close()

# Load GloVe and randomly project the vectors to 25 dimensions
glove = WordVectors.from_glove(d=50)
glove.project(np.random.randn(50, 25) * 0.0001)

# RNN training
rnn = RNN(glove, name='rnn', update=RMSprop(), step=Inverse())
rnn.train(train_trees + dev_trees, epochs=10)

# RNN prediction
pred, true = rnn.predict(test_trees)
print (true == pred.argmax(axis=1)).sum() / float(true.size)
