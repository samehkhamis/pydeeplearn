# Neural network layers (or function nodes)
# Includes an in-memory data layer, label layer, convolution, pooling, dropout, and other popular operations. Also includes cross-entropy, softmax, hinge, and squared loss layers.
# Author: Sameh Khamis (sameh@umiacs.umd.edu)
# License: GPLv2 for non-commercial research purposes only

import numpy as np
import numeric
from ..image import im2col, col2im

DTYPE = np.float32

class Node:
    def __init__(self):
        # don't save the outputs too! the circular refs make python's gc fail big
        self._input = []
        self._value = np.array([])
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        pass
    
    @property
    def input(self):
        return self._input
    
    @property
    def value(self):
        return self._value
    
    @property
    def gradient(self):
        return self._gradient
    
    def __mul__(self, other):
        return ScalarMul(self, other) if np.isscalar(other) else Mul(self, other)
    
    def __rmul__(self, other):
        return ScalarMul(self, other) if np.isscalar(other) else Mul(other, self)
    
    def __add__(self, other):
        return Add(self, other)
    
    def __radd__(self, other):
        return Add(other, self)
    
    def __pow__(self, other):
        return ScalarPow(self, other)
    
    def __neg__(self):
        return Neg(self)
    
    @property
    def T(self):
        return Trans(self)
    
    @property
    def shape(self):
        return self._value.shape
    
    @property
    def size(self):
        return self._value.size
    
    @property
    def dtype(self):
        return self._value.dtype
    
    @property
    def ndim(self):
        return self._value.ndim
    
    def __str__(self):
        return '%s [%s]' % (self.__class__.__name__, 'x'.join([str(s) for s in self.shape]))
    
    def __setstate__(self, dict):
        self.__dict__ = dict
        
        if not isinstance(self, Param):
            self.__dict__['_value'] = np.empty(self.__dict__['_value_shape'], dtype=DTYPE)
            del self.__dict__['_value_shape']
    
    def __getstate__(self):
        dict = self.__dict__.copy()
        
        del dict['_gradient']
        if '_mask' in dict: del dict['_mask']
        if '_col' in dict: del dict['_col']
        if '_temp' in dict: del dict['_temp']
        
        if isinstance(self, Label):
            del dict['_result']
        
        if not isinstance(self, Param):
            dict['_value_shape'] = self.__dict__['_value'].shape
            del dict['_value']
        
        return dict

class Op(Node):
    pass    # Base class of non-data, non-param, and non-label layers

class Data(Node):
    def __init__(self, data_mean_or_shape):
        if type(data_mean_or_shape) is tuple:
            data_mean = np.zeros(data_mean_or_shape, dtype=DTYPE)
        else:
            data_mean = data_mean_or_shape
        
        self._input = []
        self._mean = data_mean
        self._value = np.zeros(data_mean.shape, dtype=DTYPE)[np.newaxis]
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        self._value = self._value - self._mean
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)

class Preprocess(Op):
    pass    # Base class of data preprocessing (crop, contrast, tint, skew, etc.)

class Crop(Preprocess):
    def __init__(self, input, cropsize):
        self._input = [input]
        self._cropsize = cropsize
        self._value = np.empty((input.shape[0], cropsize[0], cropsize[1], input.shape[3]), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        high = np.array(self._input[0].shape[1:3]) - self._cropsize
        self._pos = np.array([np.random.randint(h) for h in high])
        self._value = self._input[0]._value[:, self._pos[0]:self._pos[0] + self._cropsize[0], self._pos[1]:self._pos[1] + self._cropsize[1], :]
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient[:, self._pos[0]:self._pos[0] + self._cropsize[0], self._pos[1]:self._pos[1] + self._cropsize[1], :] += self._gradient

class Param(Node):
    def __init__(self, val):
        self._input = []
        self._value = val.astype(DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
        self._fixed = False
    
    @staticmethod
    def zeros(shape):
        return Param(np.zeros(shape, dtype=DTYPE))
    
    @staticmethod
    def randn(shape, var=-1):
        if var < 0:
            var = np.sqrt(2.0 / np.prod(shape))
        return Param(var * np.random.randn(*shape).astype(DTYPE))

class FC(Op):
    def __init__(self, input, ndim):
        shp = input.shape
        w = Param.randn((np.prod(shp[1:]), ndim))
        b = Param.zeros((ndim,))
        self._input = [input, w, b]
        self._value = np.empty((shp[0], ndim), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        # FC = w.dot(x) + b
        data = self._input[0]._value.reshape(self._input[0].shape[0], -1)
        self._value = np.dot(data, self._input[1]._value) + self._input[2]._value
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient += self._gradient.dot(self._input[1]._value.T).reshape(self._input[0]._gradient.shape)
        data = self._input[0]._value.reshape(self._input[0].shape[0], -1)
        self._input[1]._gradient += data.T.dot(self._gradient)
        self._input[2]._gradient += self._gradient.sum(axis=0)

class Affine(FC):
    def __init__(self, input, w, b):
        shp = input.shape
        self._input = [input, w, b]
        self._value = np.empty((shp[0], b.size), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)

class Conv(Op):
    def __init__(self, input, nfilters, window=5, stride=1):
        shp = input.shape
        w = Param.randn((nfilters, window, window, shp[3]))
        b = Param.zeros((nfilters))
        self._input = [input, w, b]
        self._window = window
        self._nfilters = nfilters
        self._stride = stride
        self._value = np.empty((shp[0], shp[1], shp[2], nfilters), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
        
    def forward(self):
        n = self._input[0].shape[0]
        shp = self.shape
        
        # Reshape images to (count, channels, height, width), then apply im2col
        im = self._input[0]._value.transpose((0, 3, 1, 2))
        self._col = im2col(im, self._window, self._window, (self._window - 1) / 2, self._stride)
        
        # Now that all the windows are in matrix form, calculate w.dot(col) + b
        w = self._input[1]._value.reshape(self._nfilters, -1)
        b = self._input[2]._value.reshape(-1, 1)
        self._value = np.dot(w, self._col) + b
        
        # Reshape result from (nfilters, -1) to (count, height, width, nfilters)
        self._value = self._value.reshape(shp[3], n, shp[1], shp[2]).transpose((1, 2, 3, 0))
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        # Reshape gradient to (nfilters, -1) and back-propagate through the dot product
        gradient = self._gradient.transpose((3, 0, 1, 2)).reshape(self._nfilters, -1)
        self._input[1]._gradient += gradient.dot(self._col.T).reshape(self._input[1]._gradient.shape)
        self._input[2]._gradient += gradient.sum(axis=1)
        
        # The gradient w.r.t the images is similar, but we need to aggregate the results over the windows
        w = self._input[1]._value.reshape(self._nfilters, -1)
        shp = self._input[0].shape
        imgradient = col2im(w.T.dot(gradient), shp[0], shp[3], shp[1], shp[2], self._window, self._window, (self._window - 1) / 2, self._stride)
        
        # Reshape the result back to (count, height, width, channels)
        self._input[0]._gradient += imgradient.transpose((0, 2, 3, 1))

class Pool(Op):
    def __init__(self, input, window=2, stride=2):
        shp = input.shape
        self._input = [input]
        self._window = window
        self._stride = stride
        self._value = np.empty((shp[0], (shp[1] - window) / stride + 1, (shp[2] - window) / stride + 1, shp[3]), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        n = self._input[0].shape[0]
        shp = self.shape
        
        # Reshape images to (count, channels, height, width), then apply im2col
        im = self._input[0]._value.transpose((0, 3, 1, 2))
        col = im2col(im, self._window, self._window, 0, self._stride)
        col = col.reshape(self._window * self._window, im.shape[1], -1).transpose((1, 2, 0))
        
        # Find the maximum in every window and store its index using a mask
        self._mask = col.argmax(axis=2)
        self._mask = (self._mask[:, :, np.newaxis] == np.arange(self._window * self._window))
        self._value = col[self._mask].reshape(shp[3], n, shp[1], shp[2]).transpose((1, 2, 3, 0))
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        shp = self._input[0].shape
        
        # The gradient is calculate using the mask indices then aggregated over the windows
        gradient = self._gradient.transpose((3, 0, 1, 2)).reshape(1, shp[3], -1)
        col = (self._mask.transpose((2, 0, 1)) * gradient).reshape(-1, gradient.shape[2])
        imgradient = col2im(col, shp[0], shp[3], shp[1], shp[2], self._window, self._window, 0, self._stride)
        
        # Reshape the result back to (count, height, width, channels)
        self._input[0]._gradient += imgradient.transpose((0, 2, 3, 1))

class ScalarMul(Op):
    def __init__(self, input, scalar=1):
        self._input = [input]
        self._scalar = scalar
        self._value = np.empty((input.shape), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        self._value = self._input[0]._value * self._scalar
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient += self._gradient * self._scalar

class ScalarPow(Op):
    def __init__(self, input, scalar=1):
        self._input = [input]
        self._scalar = scalar
        self._value = np.empty((input.shape), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        self._temp = self._input[0]._value**(self._scalar - 1)
        self._value = self._temp * self._input[0]._value
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient += self._scalar * self._temp * self._gradient

class Max(Op):
    def __init__(self, input1, input2):
        assert input1.ndim == input2.ndim
        for i in np.arange(input1.ndim):
            assert input1.shape[i] == input2.shape[i]
        self._input = [input1, input2]
        self._value = np.empty((input1.shape), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        self._mask = self._input[0]._value > self._input[1]._value
        self._value = np.where(self._mask, self._input[0]._value, self._input[1]._value)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient += np.where(self._mask, self._gradient, 0)
        self._input[1]._gradient += np.where(self._mask, 0, self._gradient)

class Relu(Op):
    def __init__(self, input, leak=0.01):
        self._input = [input]
        self._value = np.empty((input.shape), dtype=DTYPE)
        self._leak = leak
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        self._mask = self._input[0]._value > 0
        self._value = np.where(self._mask, self._input[0]._value, self._leak * self._input[0]._value)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient += np.where(self._mask, self._gradient, self._leak * self._gradient)

class Dropout(Op):
    def __init__(self, input, prob=0.5):
        self._input = [input]
        self._value = np.empty((input.shape), dtype=DTYPE)
        self._prob = prob
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self, disabled=False):
        if disabled:
            self._value = self._input[0]._value
        else:
            shp = self._input[0].shape
            self._mask = np.random.rand(*shp).astype(DTYPE) < self._prob
            self._value = np.where(self._mask, self._input[0]._value / self._prob, 0)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self, disabled=False):
        if disabled:
            self._input[0]._gradient += self._gradient
        else:
            self._input[0]._gradient += np.where(self._mask, self._gradient / self._prob, 0)

class Dot(Op):
    def __init__(self, input1, input2):
        assert input1.ndim == 2 and input2.ndim == 2
        assert input1.shape[1] == input2.shape[0]
        self._input = [input1, input2]
        self._value = np.empty((input1.shape[0], input[2].shape[1]), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        self._value = np.dot(self._input[0]._value, self._input[1]._value)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient += self._gradient.dot(self._input[1]._value.T)
        self._input[1]._gradient += self._input[0]._value.T.dot(self._gradient)

class Mul(Op):
    def __init__(self, input1, input2):
        self._input = [input1, input2]
        self._value = np.empty((input1.shape), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        self._value = self._input[0]._value * self._input[1]._value
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient += self._gradient * self._input[1]._value
        self._input[1]._gradient += self._gradient * self._input[0]._value

class Add(Op):
    def __init__(self, input1, input2):
        self._input = [input1, input2]
        self._value = np.empty((input1.shape), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        self._value = self._input[0]._value + self._input[1]._value
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient += self._gradient
        self._input[1]._gradient += self._gradient

class Concat(Op):
    def __init__(self, input1, input2):
        self._input = [input1, input2]
        self._value = np.empty((input1.shape[0], input1.shape[1] + input2.shape[1]), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        self._value = np.c_[self._input[0]._value.reshape(self._input[0].shape[0], -1), self._input[1]._value.reshape(self._input[1].shape[0], -1)]
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        k = np.prod(self._input[0].shape[1:])
        self._input[0]._gradient += self._gradient[:, :k].reshape(self._input[0]._gradient.shape)
        self._input[1]._gradient += self._gradient[:, k:].reshape(self._input[1]._gradient.shape)

class Neg(Op):
    def __init__(self, input):
        self._input = [input]
        self._value = np.empty((input.shape), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        self._value = -self._input[0]._value
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient -= self._gradient
    
class Abs(Op):
    def __init__(self, input):
        self._input = [input]
        self._value = np.empty((input.shape), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        self._value = np.abs(self._input[0]._value)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient += self._gradient * np.sign(self._input[0]._value)

class Trans(Op):
    def __init__(self, input):
        self._input = [input]
        self._value = np.empty((input.shape), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        self._value = self._input[0]._value.T
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient += self._gradient.T

class Sigmoid(Op):
    def __init__(self, input):
        self._input = [input]
        self._value = np.empty((input.shape), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        self._value = numeric.sigmoid(self._input[0]._value)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient += self._value * (1 - self._value) * self._gradient

class Tanh(Op):
    def __init__(self, input):
        self._input = [input]
        self._value = np.empty((input.shape), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        self._value = 2 * numeric.sigmoid(2 * self._input[0]._value) - 1 # tanh = 2 sig(2x) - 1
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient += (1 - self._value * self._value) * self._gradient # 1 - tanh^2

class Sum(Op):
    def __init__(self, input):
        self._input = [input]
        self._value = np.empty((1,), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def forward(self):
        self._value = np.sum(self._input[0]._value).reshape(1)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient += self._gradient

class Loss(Op):
    def __init__(self, input, label):
        self._input = [input, label]
        self._value = np.empty((1,), dtype=DTYPE)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    @property
    def labels(self):
        return self._input[1]._value
    
    @property
    def result(self):
        return self._input[1]._result # result from label

class Label(Node):
    def __init__(self):
        self._input = []
        self._value = np.zeros(1, dtype=np.int32)
        self._result = np.zeros(1, dtype=np.int32)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    @property
    def result(self):
        return self._result

class Xent(Loss):
    def forward(self):
        # A stable calculation, xent(sigmoid(x)) = (1 - t) + log(1 + exp(-x))
        self._input[1]._result = numeric.sigmoid(self._input[0]._value)
        labels = self._input[1]._value
        xent = (1 - labels.astype(DTYPE)) * self._input[0]._value + numeric.log1pexp(-self._input[0]._value)
        self._value = (np.sum(xent) / DTYPE(xent.size)).reshape(1)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        diff = self._input[1]._result - self._input[1]._value
        self._input[0]._gradient += self._gradient * diff / DTYPE(diff.size)

class Softmax(Loss):
    def forward(self):
        self._input[1]._result = numeric.softmax(self._input[0]._value)
        labels = self._input[1]._value
        logz = numeric.logsumexp(self._input[0]._value)
        value = logz - self._input[0]._value[np.arange(labels.shape[0]), labels.reshape(-1)]
        self._value = (np.sum(value) / DTYPE(value.size)).reshape(1)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        diff = self._input[1]._result - numeric.onehot(self._input[1]._value, self._input[0].shape[1])
        self._input[0]._gradient += self._gradient * diff / DTYPE(diff.size)

class Hinge(Loss):
    def forward(self):
        self._input[1]._result = self._input[0]._value
        labels = self._input[1]._value
        self._target = (2 * labels - 1)
        value = 1 - self._target * self._input[0]._value
        self._mask = value > 0
        self._value = (np.where(self._mask, value, 0).sum() / DTYPE(value.size)).reshape(1)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient += self._gradient * np.where(self._mask, -self._target, 0) / DTYPE(self._target.size)

class MultiHinge(Loss):
    def forward(self):
        self._input[1]._result = self._input[0]._value
        all = np.arange(self._input[1]._value.shape[0])
        labels = self._input[1]._value
        correct = self._input[0]._value[all, labels.reshape(-1)]
        value = self._input[0]._value - correct.reshape(-1, 1) + 1     # y_delta = 1
        value[all, labels.reshape(-1)] = 0                             # t_delta = 0
        self._argmax = np.argmax(value, axis=1)
        value = value[all, self._argmax]
        self._mask = value > 0
        self._value = (np.where(self._mask, value, 0).sum() / DTYPE(value.size)).reshape(1)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        all = np.arange(self._input[1]._value.shape[0])
        labels = self._input[1]._value
        mask = np.zeros(self._input[0].shape, dtype=DTYPE)
        mask[all[self._mask], self._argmax[self._mask]] = 1
        mask[all[self._mask], labels.reshape(-1)[self._mask]] = -1
        self._input[0]._gradient += self._gradient * mask / DTYPE(mask.size)

class Squared(Loss):
    def forward(self):
        self._input[1]._result = self._input[0]._value
        self._diff = self._input[1]._value - self._input[0]._value
        self._value = (0.5 * np.sum(self._diff**2) / DTYPE(self._diff.size)).reshape(1)
        self._gradient = np.zeros(self._value.shape, dtype=DTYPE)
    
    def backward(self):
        self._input[0]._gradient += self._gradient * -self._diff / DTYPE(self._diff.size)
