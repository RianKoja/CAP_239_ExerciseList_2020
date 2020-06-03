# !/usr/bin/python
# -*- coding: latin-1 -*-
# WAVELET LIBRARY - Based on Torrence and Combo (1998)
# author: Mabel Calim Costa
# GMAO - INPE
# 23/01/2013
# reviewed 31/01/2017 for python3.6

import numpy as np
import pylab
from pylab import detrend_mean
import math

""" Translating mfiles of the Torrence and Combo to python functions
    1 - wavetest.m
    2 - wave_bases.m
    3 - wave_signif.m
    4 - chisquare_inv.m
    5 - chisquare_solve.m
"""


def nextpow2(i):
    n = 2
    while n < i:
        n = n * 2
    return n


def wave_bases(mother, k, scale, param):
    """Computes the wavelet function as a function of Fourier frequency
    used for the CWT in Fourier space (Torrence and Compo, 1998)
    -- This def is called automatically by def wavelet --

    _____________________________________________________________________
    Inputs:
    mother - a string equal to 'Morlet'
    k      - a vectorm the Fourier frequecies
    scale  - a number, the wavelet scale
    param  - the nondimensional parameter for the wavelet function

    Outputs:
    daughter       - a vector, the wavelet function
    fourier_factor - the ratio os Fourier period to scale
    coi            - a number, the cone-of-influence size at the scale
    dofmin         - a number, degrees of freedom for each point in the
                     wavelet power (Morlet = 2)

    Call function:
    daughter,fourier_factor,coi,dofmin = wave_bases(mother,k,scale,param)
    _____________________________________________________________________
    """
    n = len(k)  # length of Fourier frequencies (came from wavelet.py)
    """CAUTION : default values"""
    if (mother == 'Morlet'):  # choose the wavelet function
        param = 6  # For Morlet this is k0 (wavenumber) default is 6
        k0 = param
        # table 1 Torrence and Compo (1998)
        expnt = -pow(scale * k - k0, 2) / 2 * (k > 0)
        norm = math.sqrt(scale * k[1]) * \
            (pow(math.pi, -0.25)) * math.sqrt(len(k))
        daughter = []  # define daughter as a list

        for ex in expnt:  # for each value scale (equal to next pow of 2)
            daughter.append(norm * math.exp(ex))
        k = np.array(k)  # turn k to array
        daughter = np.array(daughter)  # transform in array
        daughter = daughter * (k > 0)  # Heaviside step function
        # scale --> Fourier
        fourier_factor = (4 * math.pi) / (k0 + math.sqrt(2 + k0 * k0))
        # cone-of- influence
        coi = fourier_factor / math.sqrt(2)
        dofmin = 2  # degrees of freedom
# ---------------------------------------------------------#
    elif (mother == 'DOG'):
        param = 2
        m = param
        expnt = -pow(scale * k, 2) / 2.0
        pws = (pow(scale * k, m))
        pws = np.array(pws)
        """CAUTION gamma(m+0.5) = 1.3293"""
        norm = math.sqrt(scale * k[1] / 1.3293) * math.sqrt(n)
        daughter = []
        for ex in expnt:
            daughter.append(-norm * pow(1j, m) * math.exp(ex))
        daughter = np.array(daughter)
        daughter = daughter[:] * pws
        fourier_factor = (2 * math.pi) / math.sqrt(m + 0.5)
        coi = fourier_factor / math.sqrt(2)
        dofmin = 1
# ---------------------------------------------------------#
    elif (mother == 'PAUL'):  # Paul Wavelet
        param = 4
        m = param
        k = np.array(k)
        expnt = -(scale * k) * (k > 0)
        norm = math.sqrt(scale * k[1]) * \
        (2 ** m / math.sqrt(m * \
                            (math.factorial(2 * m - 1)))) * math.sqrt(n)
        pws = (pow(scale * k, m))
        pws = np.array(pws)
        daughter = []
        for ex in expnt:
            daughter.append(norm * math.exp(ex))
        daughter = np.array(daughter)
        daughter = daughter[:] * pws
        daughter = daughter * (k > 0)     # Heaviside step function
        fourier_factor = 4 * math.pi / (2 * m + 1)
        coi = fourier_factor * math.sqrt(2)
        dofmin = 2
    else:
        print ('Mother must be one of MORLET,PAUL,DOG')

    return daughter, fourier_factor, coi, dofmin


def wavelet(Y, dt, param, dj, s0, j1, mother):
    """Computes the wavelet continuous transform of the vector Y,
       by definition:

    W(a,b) = sum(f(t)*psi[a,b](t) dt)        a dilate/contract
    psi[a,b](t) = 1/sqrt(a) psi(t-b/a)       b displace

    Only Morlet wavelet (k0=6) is used
    The wavelet basis is normalized to have total energy = 1 at all scales

    _____________________________________________________________________
    Input:
    Y - time series
    dt - sampling rate
    mother - the mother wavelet function
    param - the mother wavelet parameter

    Output:
    ondaleta - wavelet bases at scale 10 dt
    wave - wavelet transform of Y
    period - the vector of "Fourier"periods ( in time units) that correspond
             to the scales
    scale - the vector of scale indices, given by S0*2(j*DJ), j =0 ...J1
    coi - cone of influence

    Call function:
    ondaleta, wave, period, scale, coi = wavelet(Y,dt,mother,param)
    _____________________________________________________________________

    """

    n1 = len(Y)  # time series length
    #s0 = 2 * dt  # smallest scale of the wavelet
    # dj = 0.25  # spacing between discrete scales
    # J1 = int(np.floor((np.log10(n1*dt/s0))/np.log10(2)/dj))
    J1 = int(np.floor(np.log2(n1 * dt / s0) / dj))  # J1+1 total os scales
    # print 'Nr of Scales:', J1
    # J1= 60
    # pad if necessary
    x = detrend_mean(Y)  # extract the mean of time series
    pad = 1
    if (pad == 1):
        base2 = nextpow2(n1)  # call det nextpow2
    n = base2
    print ("n")
    """CAUTION"""
    # construct wavenumber array used in transform
    # simetric eqn 5
    #k = np.arange(n / 2)
    import math
    k_pos, k_neg = [], []
    for i in range(0, int(n / 2)):
        k_pos.append(i * ((2 * math.pi) / (n * dt)))  # frequencies as in eqn5
        k_neg = k_pos[::-1]  # inversion vector
        k_neg = [e * (-1) for e in k_neg]  # negative part
        # delete the first value of k_neg = last value of k_pos
        #k_neg = k_neg[1:-1]
    k = np.concatenate((k_pos, k_neg), axis=0)  # vector of symmetric
    # compute fft of the padded time series
    f = np.fft.fft(x, n)
    scale = []
    for i in range(J1 + 1):
        scale.append(s0 * pow(2, (i) * dj))

    period = scale
    # print period
    wave = np.zeros((J1 + 1, n))  # define wavelet array
    wave = wave + 1j * wave  # make it complex
    # loop through scales and compute transform
    for a1 in range(J1 + 1):
        daughter, fourier_factor, coi, dofmin = wave_bases(
            mother, k, scale[a1], param)  # call wave_bases
        wave[a1, :] = np.fft.ifft(f * daughter)  # wavelet transform
        if a1 == 11:
            ondaleta = daughter
    # ondaleta = daughter
    period = np.array(period)
    period = period[:] * fourier_factor

    # cone-of-influence, differ for uneven len of timeseries:
    if (((n1) / 2.0).is_integer()) is True:
        # create mirrored array)
        mat = np.concatenate(
            (arange(1,int( n1 / 2)), arange(1,int( n1 / 2))[::-1]), axis=0)
        # insert zero at the begining of the array
        mat = np.insert(mat, 0, 0)
        mat = np.append(mat, 0)  # insert zero at the end of the array
    elif (((n1) / 2.0).is_integer()) is False:
        # create mirrored array
        mat = np.concatenate(
            (arange(1,int( n1 / 2) + 1), arange(1, int(n1 / 2))[::-1]), axis=0)
        # insert zero at the begining of the array
        mat = np.insert(mat, 0, 0)
        mat = np.append(mat, 0)  # insert zero at the end of the array
    coi = [coi * dt * m for m in mat]  # create coi matrix
    # problem with first and last entry in coi added next to lines because 
    # log2 of zero is not defined and cannot be plottet later:
    coi[0] = 0.1  # coi[0] is normally 0
    coi[len(coi)-1] = 0.1 # coi[last entry] is normally 0 too
    wave = wave[:, 0:n1]
    return ondaleta, wave, period, scale, coi, f


def wave_signif(Y, dt, scale1, sigtest, lag1, sig1v1, dof, mother, param):
    """CAUTION : default values"""
    import scipy
    from scipy import stats

    n1 = np.size(Y)
    J1 = len(scale1) - 1
    s0 = np.min(scale1)
    dj = np.log10(scale1[1] / scale1[0]) / np.log10(2)
    """CAUTION"""
    if (n1 == 1):
        variance = Y
    else:
        variance = np.var(Y)
    """CAUTION"""
    # sig1v1 = 0.95
    if (mother == 'Morlet'):
        # get the appropriate parameters [see table2]
        param = 6
        k0 = param
        fourier_factor = float(4 * math.pi) / (k0 + np.sqrt(2 + k0 * k0))
        empir = [2, -1, -1, -1]
        if(k0 == 6):
            empir[1:4] = [0.776, 2.32, 0.6]

    if(mother == 'DOG'):
        param = 2
        k0 = param
        m = param
        fourier_factor = float(2 * math.pi / (np.sqrt(m + 0.5)))
        empir = [1, -1, -1, -1]
        if(k0 == 2):
            empir[1:4] = [3.541, 1.43, 1.4]

    if (mother == 'PAUL'):
        param = 4
        m = param
        fourier_factor = float(4 * math.pi / (2 * m + 1))
        empir = [2., -1, -1, -1]
        if (m == 4):
            empir[1:4] = [1.132, 1.17, 1.5]

    period = [e * fourier_factor for e in scale1]
    dofmin = empir[0]  # Degrees of  freedom with no smoothing
    Cdelta = empir[1]  # reconstruction factor
    gamma_fac = empir[2]  # time-decorrelation factor
    dj0 = empir[3]  # scale-decorrelation factor
    freq = [dt / p for p in period]
    fft_theor = [((1 - lag1 * lag1) / (1 - 2 * lag1 *
                                       np.cos(f * 2 * math.pi) + lag1 * lag1))
                 for f in freq]
    fft_theor = [variance * ft for ft in fft_theor]
    signif = fft_theor
    if(dof == -1):
        dof = dofmin
    """CAUTION"""
    if(sigtest == 0):
        dof = dofmin
        chisquare = scipy.special.gammaincinv(dof / 2.0, sig1v1) * 2.0 / dof
        signif = [ft * chisquare for ft in fft_theor]
    elif (sigtest == 1):
        """CAUTION: if len(dof) ==1"""
        dof = np.array(dof)
        truncate = np.where(dof < 1)
        dof[truncate] = np.ones(np.size(truncate))
        for i in range(len(scale1)):
            dof[i] = (
                dofmin * np.sqrt(1 + pow((dof[i] * dt / gamma_fac / scale1[i]),
                                         2)))
        dof = np.array(dof)  # has to be an array to use np.where
        truncate = np.where(dof < dofmin)
        # minimum DOF is dofmin
        dof[truncate] = [dofmin * n for n in np.ones(len(truncate))]
        chisquare, signif = [], []
        for a1 in range(J1 + 1):
            chisquare.append(
                scipy.special.gammaincinv(dof[a1] / 2.0, sig1v1) * 2.0 /
                dof[a1])
            signif.append(fft_theor[a1] * chisquare[a1])
    """CAUTION : missing elif(sigtest ==2)"""
    return signif, fft_theor
