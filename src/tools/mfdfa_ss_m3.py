#MFDFA-Analytics-by-SKDataScience
#multifractal DFA singularity spectra - module 03
#Version 3.0 - Modified by R.R.Rosa - Dec 2018 - mfdfa_ss_m3.py
# This function determines the optimal linear approximations of the data measure using two segments and returns
# the index of the corresponding boundary scale (a.k.a. crossover), the boundary scale itself, as well as the
# unifractal characteristics at the major and minor scales. For examples of using crossovers, see [1, 2].
#
# At the input, 'timeMeasure' is a time measure at different scales, while 'dataMeasure' is a data measure at the same
# scales.
#
# At the output, 'bScale' is the boundary scale, or crossover, separating the major and minor scales, 'bDM' is the
# data measure at the boundary scale, 'bsIndex' is the crossover's index with respect to the time measure, 'HMajor' is
# the unifractal dimension at the major scales, 'HMinor' is the unifractal dimension at the minor scales.
#
# REFERENCES:
# [1] D.M. Filatov, J. Stat. Phys., 165 (2016) 681-692. DOI: 10.1007/s10955-016-1641-6.
# [2] C.-K. Peng, S. Havlin, H.E. Stanley and A.L. Goldberger, Chaos, 5 (1995) 82–87. DOI: 10.1063/1.166141.
#
# The end user is granted perpetual permission to reproduce, adapt, and/or distribute this code, provided that
# an appropriate link is given to the original repository it was downloaded from.

import sys
import numpy as np


def get_scaling_exponents(time_measure, data_measure):
    # Initialisation
    n_scales = len(time_measure)
    
    log10tm = np.log10(time_measure)
    log10dm = np.log10(data_measure)
    log10dm[log10dm == -np.inf] = sys.float_info.min
    log10dm[log10dm == +np.inf] = sys.float_info.max
    
    res = 1.0e+07
    bs_index = n_scales
    
    # Computing
    # We find linear approximations for major and minor subsets of the data measure and determine the index of the
    # boundary scale at which the approximations are optimal in the sense of best fitting to the data measure
    for i in range(3, n_scales - 2 + 1):
        # Major 'i' scales are approximated by the function 'k * x + b' ...
        curr_log10tm = log10tm[n_scales - i + 1 - 1: n_scales]
        curr_log10dm = log10dm[n_scales - i + 1 - 1: n_scales]
        det_a = i * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) ** 2.0
        det_k = i * np.sum(np.multiply(curr_log10tm, curr_log10dm)) - np.sum(curr_log10tm) * np.sum(curr_log10dm)
        det_b = np.sum(curr_log10dm) * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) * np.sum(np.multiply(curr_log10tm, curr_log10dm))
        k = det_k / det_a
        b = det_b / det_a
        # ... and the maximum residual is computed
        res_major = max(np.abs(k * curr_log10tm + b - curr_log10dm))
        
        # Minor 'n_scales - i + 1' scales are approximated by the function 'k * x + b' ...
        curr_log10tm = log10tm[1 - 1: n_scales - i + 1]
        curr_log10dm = log10dm[1 - 1: n_scales - i + 1]
        det_a = (n_scales - i + 1) * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) ** 2.0
        det_k = (n_scales - i + 1) * np.sum(np.multiply(curr_log10tm, curr_log10dm)) - np.sum(curr_log10tm) * np.sum(curr_log10dm)
        det_b = np.sum(curr_log10dm) * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) * np.sum(np.multiply(curr_log10tm, curr_log10dm))
        k = det_k / det_a
        b = det_b / det_a
        # ... and the maximum residual is computed
        res_minor = max(np.abs(k * curr_log10tm + b - curr_log10dm))
        
        if res_major ** 2.0 + res_minor ** 2.0 < res:
            res = res_major ** 2.0 + res_minor ** 2.0
            bs_index = i

    # Now we determine the boundary scale and the boundary scale's data measure, ...
    b_scale = 2.0 * time_measure[1 - 1] / time_measure[n_scales - bs_index + 1 - 1] / 2.0
    b_dm = data_measure[n_scales - bs_index + 1 - 1]
    # ... as well as compute the unifractal dimensions using the boundary scale's index:
    # at the major 'bs_index' scales ...
    curr_log10tm = log10tm[n_scales - bs_index + 1 - 1: n_scales]
    curr_log10dm = log10dm[n_scales - bs_index + 1 - 1: n_scales]
    det_a = bs_index * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) ** 2.0
    det_k = bs_index * np.sum(np.multiply(curr_log10tm, curr_log10dm)) - np.sum(curr_log10tm) * np.sum(curr_log10dm)
    d_major = det_k / det_a
    h_major = -d_major
    # ... and at the minor 'n_scales - bs_index + 1' scales
    curr_log10tm = log10tm[1 - 1: n_scales - bs_index + 1]
    curr_log10dm = log10dm[1 - 1: n_scales - bs_index + 1]
    det_a = (n_scales - bs_index + 1) * np.sum(curr_log10tm ** 2.0) - np.sum(curr_log10tm) ** 2.0
    det_k = (n_scales - bs_index + 1) * np.sum(np.multiply(curr_log10tm, curr_log10dm)) - np.sum(curr_log10tm) * np.sum(curr_log10dm)
    d_minor = det_k / det_a
    h_minor = -d_minor
    
    return [b_scale, b_dm, bs_index, h_major, h_minor]
