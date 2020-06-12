########################################################################################################################
# Class to work with p-model random series generator.
#
# Adapted from https://github.com/reinaldo-rosa-inpe/cap239
#
# Adapted by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################

# P-model from Meneveau & Sreenevasan, 1987 & Malara et al., 2016
# Author: R.R.Rosa & N. Joshi
# Version: 1.6
# Date: 11/04/2018

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports:
from tools import stat


def pmodel(n_values=8192, p=0.4999, slope=None):
    no_orders = int(np.ceil(np.log2(n_values)))
    no_values_generated = 2 ** no_orders

    y = np.array([1])
    for n in range(no_orders):
        y = next_step_1d(y, p)

    if slope is not None:
        fourier_coeff = fractal_spectrum_1d(n_values, slope / 2)
        mean_val = np.mean(y)
        stdy = np.std(y)
        x = np.fft.ifft(y - mean_val)
        phase = np.angle(x)
        x = fourier_coeff * np.exp(1j * phase)
        x = np.fft.fft(x).real
        x *= stdy / np.std(x)
        x += mean_val
    else:
        x = y

    return x[0:n_values], y[0:n_values]


def next_step_1d(y, p):
    y2 = np.zeros(y.size * 2)
    sign = np.random.rand(1, y.size) - 0.5
    sign /= np.abs(sign)
    y2[0:2 * y.size:2] = y + sign * (1 - 2 * p) * y
    y2[1:2 * y.size + 1:2] = y - sign * (1 - 2 * p) * y

    return y2


def fractal_spectrum_1d(n_values, slope):
    ori_vector_size = n_values
    ori_half_size = ori_vector_size // 2
    a = np.zeros(ori_vector_size)

    for t2 in range(ori_half_size):
        index = t2
        t4 = 1 + ori_vector_size - t2
        if t4 >= ori_vector_size:
            t4 = t2
        coeff = (index + 1) ** slope
        a[t2] = coeff
        a[t4] = coeff

    a[1] = 0

    return a


class PModelGenerator:
    def __init__(self):
        self.name = "P-Model"
        self.length = 8192
        self.types_names = ["exogen", "exogen", "exogen", "endogen", "endogen", "endogen"]
        self.p = [0.20, 0.22, 0.27, 0.34, 0.38, 0.40]
        self.normalize_flg = False

    def generator(self, p, slope):
        ignore, series = pmodel(n_values=self.length, p=p, slope=slope)
        new_df = pd.DataFrame(stat.series2datasetline(series, self.normalize_flg), index=[1])
        return new_df

    def makedataframe(self, df):
        for type_name, p in zip(self.types_names, self.p):
            for trial in range(0, 20):  # Gerando 20 sinais.
                data = self.generator(p, np.random.rand() * 3)
                data['Type'] = type_name
                df = df.append(data, ignore_index=True, sort=False)
        return df


if __name__ == '__main__':

    # Endogenous (setup: N, p: 0.32-0.42, beta=0.4)
    _, yp = pmodel(n_values=256, p=0.32, slope=0.4)
    yp = yp - 1
    plt.figure()
    plt.plot(yp, marker=".")
    plt.title("Endogenous Series")
    plt.ylabel("Amplitude Values")
    plt.xlabel("Time steps")
    plt.draw()

    # Exogenous (setup: N, p: 0.18-0.28, beta=0.7)
    _, yp = pmodel(n_values=256, p=0.18, slope=0.7)
    yp = yp - 1
    plt.figure()
    plt.plot(yp, marker=".")
    plt.title("Exogenous Series")
    plt.ylabel("Amplitude Values")
    plt.xlabel("Time steps")
    plt.show()
