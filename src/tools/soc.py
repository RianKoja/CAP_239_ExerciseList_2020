# Código para SOC Power-Law

import os
import numpy as np
import matplotlib.pyplot as plt


def SOC(data, n_bins=50):
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    std = np.std(data)
    # print("mean: ", mean, " var: ", var)
    """ Computa a Taxa Local de Flutuação para cada valor da ST """
    gamma = []

    for i in range(0, n):  # gamma.append((data[i] - mean)/var)
        gamma.append((data[i] - mean) / std)

        """ Computa P[Psi_i] """
        # Retorna o número de elementos em cada bin, bem como os delimitare
    this_counts, bins = np.histogram(gamma, n_bins)
    prob_gamma = []
    for i in range(0, n_bins):
        prob_gamma.append(this_counts[i] / n)  # plt.plot(gamma)

    return prob_gamma, this_counts


# Modulo 2 ############################
if __name__ == '__main__':
    data = np.genfromtxt(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'mount', 'data.txt'))

    Prob_Gamma, counts = SOC(data)

    x = np.linspace(1, len(counts), len(counts))

    log_Prob = np.log10(Prob_Gamma)
    log_counts = np.log10(counts)

    p = np.array(Prob_Gamma)
    p = p[np.nonzero(p)]
    c = counts[np.nonzero(counts)]
    log_p = np.log10(p)
    log_c = np.log10(c)

    a = (log_p[np.argmax(c)] - log_p[np.argmin(c)]) / (np.max(c) - np.min(c))
    b = log_Prob[0]
    y = b * np.power(10, (a * counts))

    # Plotagem
    plt.clf()
    plt.scatter(np.log10(counts), y, marker=".", color="blue")

    plt.title('SOC', fontsize=16)
    plt.xlabel('log(ni)')
    plt.ylabel('log(Yi)')
    plt.grid()

    # plt.savefig('s7plot_novo.pdf')
    plt.show()

