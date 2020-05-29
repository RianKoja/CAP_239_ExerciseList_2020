import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme, norm
import seaborn as sns
from pandas.compat import BytesIO


# Example usage:
def plot_ks_gev_gauss(data_sample, alg_name):
    data_min = min(data_sample)
    data_max = max(data_sample)
    n_points = 100
    plot_points = [(data_min + (i/n_points) * (data_max-data_min)) for i in range(0, n_points+1)]

    # Estimate gaussian:
    nrm_fit = norm.fit(data_sample)
    # GEV parameters from fit:
    (mu, sigma) = nrm_fit
    rv_nrm = norm(loc=mu, scale=sigma)
    # Create data from estimated GEV to plot:
    nrm_pdf = rv_nrm.pdf(plot_points)

    # Estimate GEV:
    gev_fit = genextreme.fit(data_sample)
    # GEV parameters from fit:
    c, loc, scale = gev_fit
    rv_gev = genextreme(c, loc=loc, scale=scale)
    # Create data from estimated GEV to plot:
    gev_pdf = rv_gev.pdf(plot_points)

    # Use Kernel-Density Estimation for comparison

    # Make a Kernel density plot:
    sns.set(color_codes=True)
    plt.figure()
    ax = sns.kdeplot(data_sample, label='Kernel Density')
    ax.plot(plot_points, gev_pdf, label='Estimated GEV')
    ax.plot(plot_points, nrm_pdf, label='Estimated Gaussian')
    ax.legend()

    # Use title to indicate parameters found:
    plot_title = "PDF estimated from data created with " + alg_name + "\n"
    plot_title += "Estimated parameters for GEV: location={:.2f} scale={:.2f} c={:.2f}\n".format(loc, scale, c)
    plot_title += "Estimated parameters for Gaussian: location={:.2f} scale={:.2f}\n".format(mu, sigma)

    plt.title(plot_title)
    plt.xlabel("Independent Variable")
    plt.ylabel("Probability Density from " + str(len(data_sample)) + " points")
    plt.tight_layout()
    plt.draw()


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
    plot_ks_gev_gauss(data_white, "White Noise")
    memfile1 = BytesIO()
    plt.savefig(memfile1)
    doc.add_fig(memfile1)

    # For Red noise:
    data_red = powerlaw_psd_gaussian(2, 8192)
    plot_ks_gev_gauss(data_red, "Red Noise")
    memfile2 = BytesIO()
    plt.savefig(memfile2)
    doc.add_fig(memfile2)

if __name__ == '__main__':
    mean, cov = [0, 2], [(1, .5), (.5, 1)]
    series, y = np.random.multivariate_normal(mean, cov, size=100).T
    plot_ks_gev_gauss(series, 'example np.random')

    from tools.createdocument import ReportDocument
    doc_report = ReportDocument()
    plot_estimates_noises(doc_report)
    doc_report.finish()
    plt.show()
