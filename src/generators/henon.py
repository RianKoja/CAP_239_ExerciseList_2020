# Gerador de Mapa Logístico Caótico 2D (Henon Map): Atrator e Série Temporal
# 2D Chaotic Logistic Map Generator (Henon Map): Attractor and Time Series
# Reinaldo R. Rosa - LABAC-INPE
# Version 1.0 for CAP239-2020
# Version 2.0 modified by Rian Koja

import matplotlib.pyplot as plt


# 2D Henon logistic map is noise-like with "a" in (1.350,1.420) and "b" in (0.210,0.310)
def henon_map(a, b, x, y):
    return y + 1.0 - a * x * x, b * x


def henon_series(a, b, n, x_init=0.1, y_init=0.3):
    # Initial Condition
    x = [x_init]
    y = [y_init]
    xtemp = x_init
    ytemp = y_init

    for i in range(0, n):
        xtemp, ytemp = henon_map(a, b, xtemp, ytemp)
        x.append(xtemp)
        y.append(ytemp)

    return x, y


if __name__ == '__main__':
    # Map dependent parameters
    aa = 1.40
    bb = 0.210
    NN = 100

    xh, yh = henon_series(aa, bb, NN, x_init=0.1, y_init=0.3)
    # Plot the time series
    plt.figure()
    plt.plot(xh, yh, 'b.')
    plt.title("Henon Chaotic Attractor")
    plt.ylabel("Amplitude Values: Y")
    plt.xlabel("Amplitude Values: X")
    plt.draw()

    # Plot the time series
    plt.figure()
    plt.plot(yh)
    plt.title("Henon Chaotic Noise")
    plt.ylabel("Amplitude Values: Y")
    plt.xlabel("N Time Steps")
    plt.show()
