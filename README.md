# pydeeplearn: a Python Deep Learning Library
## Introduction
pydeeplearn is a simple deep learning library *written from scratch entirely in Python*. It is not meant to be a production-quality library (for that, check out Caffe, Theano, Mocha, Torch, or Deeplearning4j). I wrote this in my free time as an exercise and I am releasing the code for others to learn from. With that said, the codebase could definitely use more asserts and/or comments.

The design of the layers and the modularity is mostly inspired by Mocha and Theano, and the convolution approach expands the image (im2col and col2im) similar to the approach in Caffe (Yangqing Jia) and in code by Andrej Karpathy.

* Author: Sameh Khamis (sameh@umiacs.umd.edu)
* License: GPLv2 for non-commercial research purposes only
* Dependencies: Numpy 1.9.2 and Cython 0.19.1 (Run 'make' for im2col and col2im)

## Features
* Modular design
 * In-memory data (with augmentations), parameter, and label layers
 * Operations: convolution, pooling, dropout, and fully-connected
 * Non-linearities: relu, tanh, and sigmoid
 * Losses: cross-entropy, softmax, hinge, and squared
 * Gradient descent updates: vanilla, momentum, Nesterov's, Adagrad, and RMSprop
 * Step decay: fixed, inverse, and exponential
* Supports evaluation of DAG-connected networks (not just chains)
* Snapshot saving and loading
* Network visualization with Graphviz (export to dot files)
* Gradient checking through finite differences
* Numerically stable functions

## Demos
* Handwritten Digit Recognition: LeNet on MNIST, accuracy = 98.9% (LeCun et al, IEEE 1998)
 * Feed-forward convolutional neural network (CNN) with a fixed structure
* Sentiment Analysis: RNN on Movie Reviews, accuracy = 79.4% (Socher et al, EMNLP 2013)
 * Varying structure recursive neural network (RNN), built through parsing a sentence dependency tree. Word representations are initialized from a pre-trained WordVectors class (GloVe or word2vec).

## Bonus
* PTB Reader: parses dependency trees from the Penn Treebank Project
* WordVectors: parses GloVe files (Pennington et al), easy to extend to word2vec (Mikolov et al). Supports lower-dimensional (random) projections and querying analogies
