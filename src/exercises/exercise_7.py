
# Standard imports:
import os
import numpy as np
import matplotlib.pyplot as plt
import tools.mfdfa_ss as mfdfa

# Local imports:
from generators import grng
from generators import colorednoise
from generators import pmodel
from generators import logis
from generators import henon


def run(doc_report):
    names = ('GNRG', 'Color', 'P_model_025_exogen_beta04', 'logistic_rho3.88_tau1.1', 'henon_a1.38_b0.22')
    functions = (lambda: grng.time_series(2 ** np.random.randint(6, 13), 1),
                 lambda: colorednoise.powerlaw_psd_gaussian(np.random.uniform(0, 2), 8192),
                 lambda: pmodel.pmodel(8192, np.random.uniform(0.18, 0.42), 0.4)[1],
                 lambda: logis.logistic_series(np.random.uniform(3.85, 3.95), 0.5, 8192)[1],
                 lambda: henon.henon_series(np.random.uniform(1.35, 1.42), np.random.uniform(0.21, 0.31), 8192)[1])

    full_names = ('Non Gaussian Random Generator', 'Colored Noise Generator', 'P-Model', 'Logistic Map', 'Henon Map')
    sizes = (160, 120, 120, 120, 120)
    # sizes = (1, 1, 1, 1)
    columns = ['skewness', 'skewness²', 'kurtosis', 'alpha', 'beta', 'beta_theoretical']
    for (name, func, full_name, size) in zip(names, functions, full_names, sizes):
        # Generate a time series:
        data = func()
        mfdfa.main(data)


# Sample execution:
if __name__ == '__main__':
    np.random.seed(82745949)
    run(None)
    plt.show()
