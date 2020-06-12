# Código para soc_main Power-Law

import os
import numpy as np
import matplotlib.pyplot as plt


def soc_main(data, n_bins=50):
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    std = np.std(data)
    # print("mean: ", mean, " var: ", var)
    # Computa a Taxa Local de Flutuação para cada valor da ST
    gamma = []

    for i in range(0, n):  # gamma.append((data[i] - mean)/var)
        gamma.append((data[i] - mean) / std)

        """ Computa P[Psi_i] """
        # Retorna o número de elementos em cada bin, bem como os delimiters
    this_counts, bins = np.histogram(gamma, n_bins)
    prob_gamma = []
    for i in range(0, n_bins):
        prob_gamma.append(this_counts[i] / n)  # plt.plot(gamma)

    return prob_gamma, this_counts


def soc_plot(data, plot_title="Sample Self-Organized Criticality Plot"):
    prob_gamma, counts = soc_main(data)

    x = np.linspace(1, len(counts), len(counts))

    log_prob = np.log10(prob_gamma)
    log_counts = np.log10(counts)

    p = np.array(prob_gamma)
    p = p[np.nonzero(p)]
    c = counts[np.nonzero(counts)]
    log_p = np.log10(p)
    log_c = np.log10(c)

    a = (log_p[np.argmax(c)] - log_p[np.argmin(c)]) / (np.max(c) - np.min(c))
    b = log_prob[0]
    y = b * np.power(10, (a * counts))

    # Plotting:
    plt.clf()
    plt.scatter(np.log10(counts), y, marker=".", color="blue")

    plt.title(plot_title, fontsize=16)
    plt.xlabel('log(ni)')
    plt.ylabel('log(Yi)')
    plt.grid()
    plt.draw()


# sample executions:
if __name__ == '__main__':
    test_mean, test_cov = [1, -1], [(1, .5), (.5, 1)]
    test_data, _ = np.random.multivariate_normal(test_mean, test_cov, size=800).T

    soc_plot(test_data)
    plt.show()
