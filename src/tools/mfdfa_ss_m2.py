# MFDFA-Analytics-by-SKDataScience
# multifractal DFA singularity spectra - module 02
# Version 3.0 - Modified by R.R.Rosa - Dec 2018 - mfdfa_ss_m2.py
# This code implements a modification of the first-order multifractal analysis algorithm. It is based on the
# corresponding unifractal analysis technique described in [1]. It computes the Lipschitz-Holder multifractal
# singularity spectrum, as well as the minimum and maximum generalised Hurst exponents [2, 3].
#
# At the input, 'dx' is a time series of increments of the physical observable 'x(t)', of the length equal to an
# integer power of two greater than two (i.e. 4, 8, 16, 32, etc.), 'normType' is any real greater than or
# equal to one specifying the p-norm, 'isDFA' is a boolean value prescribing to use either the DFA-based algorithm or
# the standard Hurst (a.k.a. R/S) analysis, 'isNormalised' is a boolean value prescribing either to normalise the
# intermediate range-to-deviation (R/S) expression or to proceed computing without normalisation.
#
# At the output, 'timeMeasure' is the time measure of the data's support at different scales, 'dataMeasure' is
# the data measure at different scales computed for each value of the variable q-norm, 'scales' is the scales at which
# the data measure is computed, 'stats' is the structure containing MF-DFA statistics, while 'q' is the values of the
# q-norm used.
#
# Similarly to unifractal analysis (see getHurstByUpscaling()), the time measure is computed merely for an alternative
# representation of the dependence 'dataMeasure(q, scales) ~ scales ^ -tau(q)'.
#
# REFERENCES:
# [1] D.M. Filatov, J. Stat. Phys., 165 (2016) 681-692. DOI: 10.1007/s10955-016-1641-6.
# [2] J.W. Kantelhardt, Fractal and Multifractal Time Series, available at http://arxiv.org/abs/0804.0747, 2008.
# [3] J. Feder, Fractals, Plenum Press, New York, 1988.
#
# The end user is granted perpetual permission to reproduce, adapt, and/or distribute this code, provided that
# an appropriate link is given to the original repository it was downloaded from.
import sys
import numpy as np


def get_mss_by_upscaling(dx, norm_type=np.inf, is_dfa=1, is_normalised=1):
    # Some initialisation
    aux_eps = np.finfo(float).eps

    # We prepare an array of values of the variable q-norm
    aux = [-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.0001, 0.0, 0.0001, 0.5, 0.9999, 1.0, 1.0001, 2.0, 4.0, 8.0, 16.0,
           32.0]
    nq = len(aux)

    q = np.zeros((nq, 1))
    q[:, 1 - 1] = aux

    dx_len = len(dx)

    # We have to reserve the most major scale for shifts, so we divide the data
    # length by two. (As a result, the time measure starts from 2.0, not from
    # 1.0, see below.)
    dx_len = np.int(dx_len / 2)

    dx_shift = np.int(dx_len / 2)

    n_scales = np.int(np.round(np.log2(
        dx_len)))  # Number of scales involved. P.ss. We use 'round()' to prevent possible malcomputing of the logarithms
    j = 2 ** (np.arange(1, n_scales + 1) - 1) - 1

    data_measure = np.zeros((nq, n_scales))

    # Computing the data measures in different q-norms
    for ji in range(1, n_scales + 1):
        # At the scale 'j(ji)' we deal with '2 * (j(ji) + 1)' elements of the data 'dx'
        dx_k_len = 2 * (j[ji - 1] + 1)
        n = np.int(dx_len / dx_k_len)

        dx_left_shift = np.int(dx_k_len / 2)
        dx_right_shift = np.int(dx_k_len / 2)

        rr = np.zeros(n)
        ss = np.ones(n)
        for k in range(1, n + 1):
            # We get a portion of the data of the length '2*(j(ji)+1)' plus the data from the left and right boundaries
            dx_k_with_shifts = dx[(k - 1) * dx_k_len + 1 + dx_shift - dx_left_shift - 1:
                                  k * dx_k_len + dx_shift + dx_right_shift]

            # Then we perform free upscaling and, using the above-selected data (provided at the scale j = 0),
            # compute the velocities at the scale 'j(ji)'
            j_dx = np.convolve(dx_k_with_shifts, np.ones(dx_right_shift), 'valid')

            # Then we compute the accelerations at the scale 'j(ji) + 1'
            r = (j_dx[1 + dx_right_shift - 1:] - j_dx[1 - 1: -dx_right_shift]) / 2.0

            # Finally we compute the range ...
            if norm_type == 0:
                rr[k - 1] = np.max(r[2 - 1:]) - np.min(r[2 - 1:])
            elif np.isinf(norm_type):
                rr[k - 1] = np.max(np.abs(r[2 - 1:]))
            else:
                rr[k - 1] = (np.sum(r[2 - 1:] ** norm_type) / len(r[2 - 1:])) ** (1.0 / norm_type)
            # ... and the normalisation factor ("standard deviation")
            if is_dfa == 0:
                ss[k - 1] = np.sqrt(np.sum(np.abs(np.diff(r)) ** 2.0) / (len(r) - 1))

        if is_normalised == 1:  # Then we either normalise the rr / ss values, treating them as probabilities ...
            p = np.divide(rr, ss) / np.sum(np.divide(rr, ss))
        else:  # ... or leave them unnormalised ...
            p = np.divide(rr, ss)
        # ... and compute the measures in the q-norms
        for k in range(1, n + 1):
            # This 'if' is needed to prevent measure blow-ups with
            # negative values of 'q' when the probability is close to zero
            if p[k - 1] < 1000.0 * aux_eps:
                continue

            data_measure[:, ji - 1] = data_measure[:, ji - 1] + np.power(p[k - 1], q[:, 1 - 1])

    # We pass from the scales ('j') to the time measure; the time measure at the scale j(n_scales) (the most major one)
    # is assumed to be 2.0, while it is growing when the scale is tending to j(1) (the most minor one).
    # (The scale j(n_scales)'s time measure is NOT equal to 1.0, because we reserved the highest scale for shifts
    # in the very beginning of the function.)
    time_measure = 2.0 * dx_len / (2 * (j + 1))

    scales = j + 1

    # Determining the exponents 'tau' from 'data_measure(q, time_measure) ~ time_measure ^ tau(q)'
    tau = np.zeros((nq, 1))
    log10tm = np.log10(time_measure)
    log10dm = np.log10(data_measure)
    log10dm[log10dm == -np.inf] = sys.float_info.min
    log10dm[log10dm == +np.inf] = sys.float_info.max
    log10tm_mean = np.mean(log10tm)

    # For each value of the q-norm we compute the mean 'tau' over all the scales
    for qi in range(1, nq + 1):
        tau[qi - 1, 1 - 1] = np.sum(np.multiply(log10tm, (log10dm[qi - 1, :] - np.mean(log10dm[qi - 1, :])))) / np.sum(
            np.multiply(log10tm, (log10tm - log10tm_mean)))

    # Finally, we only have to pass from 'tau(q)' to its conjugate function 'f(alpha)'
    # In doing so, first we find the Lipschitz-Holder exponents 'alpha' (represented by the variable 'LH') ...
    aux_top = (tau[2 - 1] - tau[1 - 1]) / (q[2 - 1] - q[1 - 1])
    aux_middle = np.divide(tau[3 - 1:, 1 - 1] - tau[1 - 1: -1 - 1, 1 - 1], q[3 - 1:, 1 - 1] - q[1 - 1: -1 - 1, 1 - 1])
    aux_bottom = (tau[-1] - tau[-1 - 1]) / (q[-1] - q[-1 - 1])
    lh = np.zeros((nq, 1))
    lh[:, 1 - 1] = -np.concatenate((aux_top, aux_middle, aux_bottom))
    # ... and then compute the conjugate function 'f(alpha)' itself
    f = np.multiply(lh, q) + tau

    # The last preparations
    # We determine the minimum and maximum values of 'alpha' ...
    lh_min = lh[-1, 1 - 1]
    lh_max = lh[1 - 1, 1 - 1]
    # ... and find the minimum and maximum values of another multifractal characteristic, the so-called
    # generalised Hurst (or DFA) exponent 'h'. (These parameters are computed according to [2, p. 27].)
    h_min = -(1.0 + tau[-1, 1 - 1]) / q[-1, 1 - 1]
    h_max = -(1.0 + tau[1 - 1, 1 - 1]) / q[1 - 1, 1 - 1]

    stats = {'tau': tau,
             'LH': lh,
             'f': f,
             'LH_min': lh_min,
             'LH_max': lh_max,
             'h_min': h_min,
             'h_max': h_max}

    return [time_measure, data_measure, scales, stats, q]
