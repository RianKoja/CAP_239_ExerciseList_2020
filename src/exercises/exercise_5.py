
# Standard imports:
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from pandas.compat import BytesIO

# Local imports:
from generators import henon
from generators import logis
from tools import cullen_frey_giovanni


def plot_henon(n_points=512):
    # Generate Henon series:
    a_list = np.linspace(1.350, 1.400, num=3, endpoint=True)
    b_list = np.linspace(0.210, 0.310, num=3)
    henon_skew = []
    henon_kurt = []

    for a in a_list:
        for b in b_list:
            data_x, _ = henon.henon_series(a, b, n_points, x_init=0.1, y_init=0.3)
            henon_skew.append(skew(data_x)**2)
            henon_kurt.append(kurtosis(data_x, fisher=False))

    cullen_frey_giovanni.cullenfrey(henon_skew, henon_kurt, 'Henon', "a in " + str(a_list) + " b in " + str(b_list))


def plot_logis(n_points=512):
    # Generate Logistic map series:
    rho_list = np.linspace(3.85, 4.0, num=6)
    logis_skew = []
    logis_kurt = []
    tau = 1.1
    for rho in rho_list:
        _, data = logis.logistic_series(rho, tau, n_points)
        logis_skew.append(skew(data) ** 2)
        logis_kurt.append(kurtosis(data, fisher=False))
    cullen_frey_giovanni.cullenfrey(logis_skew, logis_kurt, 'Logistic Map', "rho in " + str(rho_list))


def report_ex5(doc_report):
    doc_report.document.add_heading('Exercise 4', level=2)
    plot_henon()
    memfile_hn = BytesIO()
    plt.savefig(memfile_hn)
    doc_report.add_fig(memfile_hn)

    plot_logis()
    memfile_lg = BytesIO()
    plt.savefig(memfile_lg)
    doc_report.add_fig(memfile_lg)


# Sample execution
if __name__ == '__main__':
    plot_henon()
    plot_logis()
    plt.show()
