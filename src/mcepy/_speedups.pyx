# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def embeddings_to_condensed(cnp.ndarray[cnp.float64_t, ndim=3] embeddings, bint normalize=True):
    cdef Py_ssize_t m = embeddings.shape[0]
    cdef Py_ssize_t n = embeddings.shape[1]
    cdef Py_ssize_t p = embeddings.shape[2]
    cdef Py_ssize_t length = n * (n - 1) // 2
    cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.empty((m, length), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] work = np.empty((n, p), dtype=np.float64)
    cdef Py_ssize_t emb_idx
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t pos
    cdef double value
    cdef double norm_sq
    cdef double inv_norm
    cdef double mean
    cdef double diff
    cdef double dist_sq

    for emb_idx in range(m):
        for i in range(n):
            for k in range(p):
                work[i, k] = embeddings[emb_idx, i, k]

        if normalize:
            for k in range(p):
                mean = 0.0
                for i in range(n):
                    mean += work[i, k]
                mean /= n
                for i in range(n):
                    work[i, k] -= mean

            norm_sq = 0.0
            for i in range(n):
                for k in range(p):
                    value = work[i, k]
                    norm_sq += value * value

            if norm_sq > 0.0:
                inv_norm = 1.0 / sqrt(norm_sq)
                for i in range(n):
                    for k in range(p):
                        work[i, k] *= inv_norm

        pos = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                dist_sq = 0.0
                for k in range(p):
                    diff = work[i, k] - work[j, k]
                    dist_sq += diff * diff
                out[emb_idx, pos] = sqrt(dist_sq)
                pos += 1

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def geometric_median_condensed(
    cnp.ndarray[cnp.float64_t, ndim=2] condensed,
    double eps=1e-15,
    double tol=1e-5,
    int max_iter=500,
):
    cdef Py_ssize_t m = condensed.shape[0]
    cdef Py_ssize_t length = condensed.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] center = np.zeros(length, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] new_center = np.zeros(length, dtype=np.float64)
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef int iteration
    cdef double value
    cdef double diff
    cdef double dist_sq
    cdef double distance
    cdef double weight
    cdef double weight_sum
    cdef double delta_sq
    cdef double objective = 0.0
    cdef bint converged = False

    if m == 0:
        raise ValueError("condensed must contain at least one embedding")

    for j in range(length):
        value = 0.0
        for i in range(m):
            value += condensed[i, j]
        center[j] = value / m

    for iteration in range(max_iter):
        for j in range(length):
            new_center[j] = 0.0

        weight_sum = 0.0
        objective = 0.0

        for i in range(m):
            dist_sq = 0.0
            for j in range(length):
                diff = condensed[i, j] - center[j]
                dist_sq += diff * diff
            distance = sqrt(2.0 * dist_sq)
            objective += distance
            weight = 1.0 / (distance + eps)
            weight_sum += weight
            for j in range(length):
                new_center[j] += weight * condensed[i, j]

        if weight_sum == 0.0:
            break

        for j in range(length):
            new_center[j] /= weight_sum

        delta_sq = 0.0
        for j in range(length):
            diff = new_center[j] - center[j]
            delta_sq += diff * diff
            center[j] = new_center[j]

        if sqrt(2.0 * delta_sq) < tol:
            converged = True
            objective /= m
            return center, iteration + 1, converged, objective

    objective = 0.0
    for i in range(m):
        dist_sq = 0.0
        for j in range(length):
            diff = condensed[i, j] - center[j]
            dist_sq += diff * diff
        objective += sqrt(2.0 * dist_sq)
    objective /= m

    return center, max_iter, converged, objective


@cython.boundscheck(False)
@cython.wraparound(False)
def condensed_to_square(cnp.ndarray[cnp.float64_t, ndim=1] condensed, Py_ssize_t n_samples):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.zeros((n_samples, n_samples), dtype=np.float64)
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t pos = 0

    for i in range(n_samples - 1):
        for j in range(i + 1, n_samples):
            out[i, j] = condensed[pos]
            out[j, i] = condensed[pos]
            pos += 1

    return out
