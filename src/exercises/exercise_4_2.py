
# Standard imports:
import numpy as np
import matplotlib.pyplot as plt
from pandas.compat import BytesIO


# Local imports:
from tools import stat


def plot_estimates_noises(doc):
    from generators.colorednoise import powerlaw_psd_gaussian
    doc.document.add_heading('Exercise 4.2', level=2)
    doc.document.add_paragraph("""
    Here we compare different methods of estimating a probability density function from sampled data.
    The first is to use Kernel Density Estimation, a standard method to compose a PDF as a pondered sum of PDF's centered at each observed point.
    The second is a parametric estimation, done for both Gaussian distribution and Generalized Extreme Value distribution. With this technique, a best parametric fit for the observed data is found, and the curve is plotted based on the parameters found.
    For each kind of noise, the type is indicated in the plot title, along with the values estimated for parameters.
    """)

    # For white noise:
    data_white = powerlaw_psd_gaussian(0, 8192)
    stat.plot_ks_gev_gauss(data_white, "White Noise")
    memfile1 = BytesIO()
    plt.savefig(memfile1)
    doc.add_fig(memfile1)

    # For Red noise:
    data_red = powerlaw_psd_gaussian(2, 8192)
    stat.plot_ks_gev_gauss(data_red, "Red Noise")
    memfile2 = BytesIO()
    plt.savefig(memfile2)
    doc.add_fig(memfile2)


# Sample execution:
if __name__ == '__main__':
    mean, cov = [0, 2], [(1, .5), (.5, 1)]
    series, y = np.random.multivariate_normal(mean, cov, size=100).T
    stat.plot_ks_gev_gauss(series, 'example np.random')

    from tools.createdocument import ReportDocument
    doc_report = ReportDocument()
    plot_estimates_noises(doc_report)
    doc_report.finish()
    plt.show()
