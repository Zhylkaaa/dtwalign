# -*- coding: utf-8 -*-
"""Cost matrix computation."""

import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def _calc_cumsum_matrix_jit(X, w_list, p_ar, open_begin):
    """Fast implementation by numba.jit."""
    batch_size, len_x, len_y = X.shape
    # cumsum matrix
    D = np.ones((batch_size, len_x, len_y), dtype=np.float32) * np.inf
    D_dir = np.full_like(D, -1, dtype=np.int32)

    if open_begin:
        X = np.vstack((np.zeros((batch_size, 1, X.shape[2])), X))
        D = np.vstack((np.zeros((batch_size, 1, D.shape[2])), D))
        w_list[:, 0] += 1

    # number of patterns
    num_pattern = p_ar.shape[0]
    # max pattern length
    max_pattern_len = p_ar.shape[1]
    # pattern cost
    pattern_cost = np.zeros((batch_size, num_pattern), dtype=np.float32)
    # step cost
    step_cost = np.zeros((batch_size, max_pattern_len), dtype=np.float32)
    # number of cells
    num_cells = w_list.shape[0]
    D[:, 0, 0] = X[:, 0, 0]

    for tidx in prange(batch_size):
        for cell_idx in range(1, num_cells):
            i = w_list[cell_idx, 0]
            j = w_list[cell_idx, 1]

            for pidx in range(num_pattern):
                # calculate local cost for each pattern
                for sidx in range(1, max_pattern_len):
                    # calculate step cost of pair-wise cost matrix
                    pattern_index = p_ar[pidx, sidx, 0:2]
                    ii = int(i + pattern_index[0])
                    jj = int(j + pattern_index[1])
                    if ii < 0 or jj < 0:
                        step_cost[tidx, sidx] = np.inf
                        continue
                    else:
                        step_cost[tidx, sidx] = X[tidx, ii, jj] \
                            * p_ar[pidx, sidx, 2]

                pattern_index = p_ar[pidx, 0, 0:2]
                ii = int(i + pattern_index[0])
                jj = int(j + pattern_index[1])
                if ii < 0 or jj < 0:
                    pattern_cost[tidx, pidx] = np.inf
                    continue

                pattern_cost[tidx, pidx] = D[tidx, ii, jj] \
                    + step_cost[tidx].sum()

            min_step = np.argmin(pattern_cost[tidx])
            min_cost = pattern_cost[tidx, min_step]
            if min_cost != np.inf:
                D[tidx, i, j] = min_cost
                D_dir[tidx, i, j] = min_step

    return D, D_dir
