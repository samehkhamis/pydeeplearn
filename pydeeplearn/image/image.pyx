# Cythonized functions: im2col, col2im, transform, invtransform
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

@cython.boundscheck(False)
@cython.wraparound(False)
def transform(np.ndarray[DTYPE_t, ndim=4] im, np.ndarray[DTYPE_t, ndim=2] A):
    cdef int N = im.shape[0], H = im.shape[1], W = im.shape[2], C = im.shape[3]
    
    cdef np.ndarray[DTYPE_t, ndim=4] transformed = np.empty((N, H, W, C), dtype=DTYPE)
    cdef int i, hj, wj, hk, wk, ci, hi, wi
    cdef DTYPE_t hin, win, alphaw, alphah
    
    for hi in range(H):
        for wi in range(W):
            hin, win = DTYPE(hi) - H / 2.0, DTYPE(wi) - W / 2.0
            alphah = A[0, 0] * hin + A[0, 1] * win + H / 2.0
            alphaw = A[1, 0] * hin + A[1, 1] * win + W / 2.0
            
            hj, wj = int(alphah), int(alphaw)
            hk, wk = hj + 1, wj + 1
            alphah, alphaw = alphah - hj, alphaw - wj
            
            hj = 0 if hj < 0 else H - 1 if hj >= H else hj
            wj = 0 if wj < 0 else W - 1 if wj >= W else wj
            hk = 0 if hk < 0 else H - 1 if hk >= H else hk
            wk = 0 if wk < 0 else W - 1 if wk >= W else wk
            
            for i in range(N):
                for ci in range(C):
                    transformed[i, hi, wi, ci] = alphah * alphaw * im[i, hj, wj, ci] +\
                                                 alphah * (1 - alphaw) * im[i, hj, wk, ci] +\
                                                 (1 - alphah) * alphaw * im[i, hk, wj, ci] +\
                                                 (1 - alphah) * (1 - alphaw) * im[i, hk, wk, ci]
    return transformed

@cython.boundscheck(False)
@cython.wraparound(False)
def invtransform(np.ndarray[DTYPE_t, ndim=4] transformed, np.ndarray[DTYPE_t, ndim=2] A):
    cdef int N = transformed.shape[0], H = transformed.shape[1], W = transformed.shape[2], C = transformed.shape[3]
    
    cdef np.ndarray[DTYPE_t, ndim=4] im = np.zeros((N, H, W, C), dtype=DTYPE)
    cdef int i, hj, wj, hk, wk, ci, hi, wi
    cdef DTYPE_t hin, win, alphaw, alphah
    
    for hi in range(H):
        for wi in range(W):
            hin, win = DTYPE(hi) - H / 2.0, DTYPE(wi) - W / 2.0
            alphah = A[0, 0] * hin + A[0, 1] * win + H / 2.0
            alphaw = A[1, 0] * hin + A[1, 1] * win + W / 2.0
            
            hj, wj = int(alphah), int(alphaw)
            hk, wk = hj + 1, wj + 1
            alphah, alphaw = alphah - hj, alphaw - wj
            
            if hj >= 0 and hj < H and wj >= 0 and wj < W:
                hk = 0 if hk < 0 else H - 1 if hk >= H else hk
                wk = 0 if wk < 0 else W - 1 if wk >= W else wk
                
                for i in range(N):
                    for ci in range(C):
                        im[i, hj, wj, ci] += alphah * alphaw * transformed[i, hi, wi, ci]
                        im[i, hj, wk, ci] += alphah * (1 - alphaw) * transformed[i, hi, wi, ci]
                        im[i, hk, wj, ci] += (1 - alphah) * alphaw * transformed[i, hi, wi, ci]
                        im[i, hk, wk, ci] += (1 - alphah) * (1 - alphaw) * transformed[i, hi, wi, ci]
    return im
