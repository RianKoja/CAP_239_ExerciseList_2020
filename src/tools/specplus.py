########################################################################################################################
# Compute and plot Detrended Fluctuation Analysis
#
# Adapted from:
# https://github.com/reinaldo-rosa-inpe/cap239/blob/f92f84710bb8438e461d2bf5d237e93bb7238d3e/Codigos/Specplus.py
# Written by Paulo Giovani, Reinaldo Roberto Rosa, Murilo da Silva Dantas
#
# Adapted by Rian Koja to publish in a GitHub repository with GPL licence for this specific file.
########################################################################################################################

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats, optimize
import numpy as np
import math


# ---------------------------------------------------------------------
# Computes PSD of a time series
# Optionally receives an interval for the linear regression step
# ---------------------------------------------------------------------
def psd(data, init=None, final=None):
    n = len(data)
    time = np.arange(n)

    # If "final" is not given, use length of data
    if final is None:
        final = n - 1
    if init is None:
        init = 1

    # Define sampling frequency:
    dt = (time[-1] - time[0] / (n - 1))
    fs = 1 / dt

    # Compute PSD with MLAB:
    power, freqs = mlab.psd(data, Fs=fs, NFFT=n, scale_by_freq=False)

    # Select data within selection interval
    xdata = freqs[init:final]
    ydata = power[init:final]

    # Simulate error
    yerr = 0.2 * ydata

    # Find logarithms of data:
    logx = np.log10(xdata)
    logy = np.log10(ydata)

    logyerr = yerr / ydata

    # Compute line fit:
    pinit = np.array([1.0, -1.0])
    out = optimize.leastsq(errfunc, pinit, args=(logx, logy, logyerr), full_output=True)
    pfinal = out[0]
    index = pfinal[1]
    amp = 10.0 ** pfinal[0]

    # Returns obtained values
    return freqs, power, xdata, ydata, amp, index, init, final


# Define functions to perform data fit:
def fitfunc(p, x):
    return p[0] + p[1] * x


def errfunc(p, x, y, err):
    return (y - fitfunc(p, x)) / err


# Define a function to compute a Power Law:
def powerlaw(x, amp, index):
    return amp * (x ** index)


# Compute 1D DFA for the time series
def dfa1d(time_series, grau):
    # Compute 1D DFA (adapted from Physionet), where the scale frowns according to 'boxratio'. Returns the array
    # 'vetoutput', where the first column is the logarithm of S scale and the second column is the logarithm of the
    # fluctuation function

    # 1. The time series {Xk} with k = 1, ..., N is integrated into the profile function Y(k)
    x = np.mean(time_series)
    time_series = time_series - x
    yk = np.cumsum(time_series)
    tam = len(time_series)

    # 2. The (or profile) Y(k) is divided into N non-overlapping intervals of size S
    sf = np.ceil(tam / 4).astype(np.int)
    boxratio = np.power(2.0, 1.0 / 8.0)
    vetoutput = np.zeros(shape=(1, 2))
    s = 4

    while s <= sf:
        serie = yk
        if np.mod(tam, s) != 0:
            aux = s * int(np.trunc(tam / s))
            serie = yk[0:aux]
        t = np.arange(s, len(serie), s)
        v = np.array(np.array_split(serie, t))
        x = np.arange(1, s + 1)

        # 3. Compute variance for each segment v = 1,…, n_s:
        p = np.polynomial.polynomial.polyfit(x, v.T, grau)
        yfit = np.polynomial.polynomial.polyval(x, p)
        vetvar = np.var(v - yfit)

        # 4. Compute the the fluctuation function of the DFA as the average of the variances in each interval
        fs = np.sqrt(np.mean(vetvar))
        vetoutput = np.vstack((vetoutput, [s, fs]))

        # S scale grows in geometric series
        s = np.ceil(s * boxratio).astype(np.int)

    # Array with S scale log values and fluctuation function log values
    vetoutput = np.log10(vetoutput[1::1, :])

    # Split the columns of 'vetoutput'
    x = vetoutput[:, 0]
    y = vetoutput[:, 1]

    # Linear Regression
    slope, intercept, _, _, _ = stats.linregress(x, y)

    # Compute line
    predict_y = intercept + slope * x

    # Compute error
    pred_error = y - predict_y
    # Returns the alpha value (slope), the 'vetoutput' vector, the X and Y vectors,
    # a vector with line values, and the error vector
    return slope, vetoutput, x, y, predict_y, pred_error


# ---------------------------------------------------------------------
# Main section
# ---------------------------------------------------------------------
def main(data):
    # Disable numpy errors and warnings
    np.seterr(divide='ignore', invalid='ignore', over='ignore')

    # -----------------------------------------------------------------
    # General plot parameters:
    # -----------------------------------------------------------------

    # Define subplots
    fig = plt.figure()
    fig.subplots_adjust(hspace=.3, wspace=.2)

    # Font sizes:
    size_font_axis_x = 10
    size_font_axis_y = 10
    size_font_title = 8
    size_font_main = 15

    # Main title
    title_main = 'Time Series Spectral Analysis'

    # -----------------------------------------------------------------
    # Plot original series
    # -----------------------------------------------------------------

    # Define plot colors
    cor_serie_original = 'r'
    # Titles for axes on the original series plot
    text_axis_x = 'Time'
    text_axis_y = 'Amplitude'
    text_title_original = 'Original Time Series Data'

    # Plot original data series:
    fig_handle = fig.add_subplot(2, 1, 1)
    fig_handle.plot(data, '-', color=cor_serie_original)
    fig_handle.set_title(text_title_original, fontsize=size_font_title)
    fig_handle.set_xlabel(text_axis_x, fontsize=size_font_axis_x)
    fig_handle.set_ylabel(text_axis_y, fontsize=size_font_axis_y)
    fig_handle.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig_handle.grid()

    # -----------------------------------------------------------------
    # Compute and plot PSD
    # -----------------------------------------------------------------

    # Compute PSD
    freqs, power, xdata, ydata, amp, index, init, fim = psd(data)

    # Beta value is equivalent to the index:
    beta = index

    # Define plot colors:
    cor_psd1 = 'k'
    cor_psd2 = 'navy'

    # PSD axes titles:
    texto_psdx = 'Frequency (Hz)'
    texto_psdy = 'Power'
    texto_titulo_psd = r'Power Spectrum Density $\beta$ = '

    # -----------------------------------------------------------------
    # Compute and plot DFA
    # -----------------------------------------------------------------

    # Compute 1D DFA
    alfa, vetoutput, x, y, reta, error = dfa1d(data, 1)
    # From Neelakshi et. al. (2019) "Spectral fluctuation analysis of ionospheric inhomogeneities over Brazilian
    # territory Part II: EF valley region plasma instabilities"
    # "S. Heneghan and McDarby (2000) established an equivalence relation between the PSD exponent, b, and the DFA
    # exponent, a, given by beta =  2 * alpha - 1. Kiyono (2015) showed that this relationship is valid for
    # the higher order DFA subject to the constraint  0<a<m+1, where m is the order of detrending polynomial in the DFA"
    beta_theoretical = 2 * alfa - 1

    # Plot PSD

    fig_handle = fig.add_subplot(2, 2, 3)

    fig_handle.plot(freqs, power, '-', color=cor_psd1, alpha=0.7)
    fig_handle.plot(xdata, ydata, color=cor_psd2, alpha=0.8)
    fig_handle.axvline(freqs[init], color=cor_psd2, linestyle='--')
    fig_handle.axvline(freqs[-1], color=cor_psd2, linestyle='--')
    fig_handle.plot(xdata, powerlaw(xdata, amp, index), 'r-', linewidth=1.5, label='$%.4f$' % beta)
    fig_handle.set_xlabel(texto_psdx, fontsize=size_font_axis_x)
    fig_handle.set_ylabel(texto_psdy, fontsize=size_font_axis_y)
    fig_handle.set_title(texto_titulo_psd + r'%.4f (Theoretical $\beta$ = 2$\alpha$ -1 = %.4f)' % (beta,
                                                                                                   beta_theoretical),
                         loc='center', fontsize=size_font_title)
    fig_handle.set_yscale('log')
    fig_handle.set_xscale('log')
    fig_handle.grid()

    # Checks if DFA has valid value. If so, proceed with plot:

    if not math.isnan(alfa):

        # Define plot colors:
        cor_dfa = 'darkmagenta'

        # DFA axes title:
        texto_dfax = '$log_{10}$ (s)'
        texto_dfay = '$log_{10}$ F(s)'
        texto_titulo_dfa = r'Detrended Fluctuation Analysis $\alpha$ = '

        # Plot DFA
        fig_dfa = fig.add_subplot(2, 2, 4)
        fig_dfa.plot(x, y, 's', color=cor_dfa, markersize=4, markeredgecolor='r', markerfacecolor='None', alpha=0.8)
        fig_dfa.plot(x, reta, '-', color=cor_dfa, linewidth=1.5)
        fig_dfa.set_title(texto_titulo_dfa + '%.4f' % alfa, loc='center', fontsize=size_font_title)
        fig_dfa.set_xlabel(texto_dfax, fontsize=size_font_axis_x)
        fig_dfa.set_ylabel(texto_dfay, fontsize=size_font_axis_y)
        fig_dfa.grid()

    else:
        fig_dfa = fig.add_subplot(2, 2, 4)
        fig_dfa.set_title('Detrended Fluctuation Analysis $\alpha$ = ' + 'N.A.', loc='center',
                          fontsize=size_font_title)
        fig_dfa.grid()

    # Draw and save figure (avoid showing to prevent blocking)
    plt.suptitle(title_main, fontsize=size_font_main)
    fig.set_size_inches(10, 5)
    # img_filename = 'ANALYSIS_PSD_DFA_2.png'
    # plt.savefig(img_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.draw()

    return alfa, beta_theoretical, beta


# Sample execution:
if __name__ == "__main__":
    mean, cov = [1, -1], [(1, .5), (.5, 1)]
    series_x, series_y = np.random.multivariate_normal(mean, cov, size=800).T
    test_data = series_x.tolist()
    alpha, beta_t, beta_o = main(test_data)
    plt.show()
