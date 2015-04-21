# Cythonized imcol and colim
# Started from Andrej Karpathy's code, made faster by avoiding np.pad
# Author: Sameh Khamis (sameh@umiacs.umd.edu)
# License: GPLv2 for non-commercial research purposes only

import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def im2col(np.ndarray[DTYPE_t, ndim=4] im, int filterH, int filterW, int padding, int stride):
    cdef int N = im.shape[0], C = im.shape[1], H = im.shape[2], W = im.shape[3]
    cdef np.ndarray[DTYPE_t, ndim=4] im_padded = np.empty((N, C, H + 2 * padding, W + 2 * padding), dtype=DTYPE)
    if padding > 0:
        im_padded[:, :, padding:-padding, padding:-padding] = im
        im_padded[:, :, :padding, :] = 0
        im_padded[:, :, -padding:, :] = 0
        im_padded[:, :, :, :padding] = 0
        im_padded[:, :, :, -padding:] = 0
    else:
        im_padded[:] = im
    
    cdef int newH = (H + 2 * padding - filterH) / stride + 1
    cdef int newW = (W + 2 * padding - filterW) / stride + 1
    
    cdef np.ndarray[DTYPE_t, ndim=2] col = np.empty((C * filterH * filterW, N * newH * newW), dtype=DTYPE)
    cdef int i, hj, wj, ci, hi, wi, c, r
    
    for hi in range(filterH):
        for wi in range(filterW):
            for ci in range(C):
                r = hi * filterW * C + wi * C + ci
                for i in range(N):
                    for hj in range(newH):
                        for wj in range(newW):
                            c = i * newH * newH + hj * newW + wj
                            col[r, c] = im_padded[i, ci, stride * hj + hi, stride * wj + wi]
    return col

@cython.boundscheck(False)
@cython.wraparound(False)
def col2im(np.ndarray[DTYPE_t, ndim=2] col, int N, int C, int H, int W,
        int filterH, int filterW, int padding, int stride):
    cdef int newH = (H + 2 * padding - filterH) / stride + 1
    cdef int newW = (W + 2 * padding - filterW) / stride + 1
    
    cdef np.ndarray[DTYPE_t, ndim=4] im_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=DTYPE)
    cdef int i, hj, wj, ci, hi, wi, c, r
    
    for ci in range(C):
        for hi in range(filterH):
            for wi in range(filterW):
                r = hi * filterW * C + wi * C + ci
                for i in range(N):
                    for hj in range(newH):
                        for wj in range(newW):
                            c = i * newH * newH + hj * newW + wj
                            im_padded[i, ci, stride * hj + hi, stride * wj + wi] += col[r, c]
    
    if padding > 0:
        return im_padded[:, :, padding:-padding, padding:-padding]
    return im_padded
    