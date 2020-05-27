# Gerador de Mapa Logístico Caótico 1D: Atrator e Série Temporal
# 1D Chaotic Logistic Map Generator: Attractor and Time Series
# Reinaldo R. Rosa - LABAC-INPE
# Version 1.0 for CAP239-2020
# Version 2.0 adapted by Rian Koja

import matplotlib.pyplot as plt


# chaotic logistic map is f(x) = rho*x*(1-x)  with rho in (3.81,4.00)
def logistic(rho, tau, x, y):
    return rho*x*(1.0-x), tau*x


def logistic_series(rho, tau, n_points, x_init=0.001, y_init=0.001):
    xtemp = x_init
    ytemp = y_init
    x = [xtemp]
    y = [ytemp]

    for i in range(1, n_points):
        xtemp, ytemp = logistic(rho, tau, xtemp, ytemp)
        x.append(xtemp)
        y.append(ytemp)

    return x, y


# Sample run:
if __name__ == '__main__':
    # Map dependent parameters
    test_rho = 3.88
    test_tau = 1.1
    n = 256

    xlog, ylog = logistic_series(test_rho, test_tau, n, x_init=0.001, y_init=0.001)

    # Plot the Attractor
    plt.figure()
    plt.plot(xlog, ylog, 'b.')
    plt.title("Logistic Chaotic Attractor")
    plt.ylabel("Amplitude Values: A(t)")
    plt.xlabel("Amplitude Values: A(t+tau)")
    plt.draw()

    # Plot the time series
    plt.figure()
    plt.plot(xlog)
    plt.title("Logistic Chaotic Noise")
    plt.ylabel("Amplitude Values: A(t)")
    plt.xlabel("Time step")
    plt.show()

