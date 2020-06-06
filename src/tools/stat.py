########################################################################################################################
# Used to compute statisticas parameters of data, used in several exercises, such as skewness and kurtosis.
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################


# Standard imports:
from statistics import mean, variance
from scipy.stats import skew, kurtosis, genextreme, norm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns


# This functions takes a series of data, linearly interpolates it to the [0, 1] interval,
# Then computes its sample statistical moments, returns it in dictionary form to be appended in a data frame.
def series2datasetline(series, normalize):
    if normalize:  # Normalize to the [0, 1] interval:
        mapper = interp1d([min(series), max(series)], [0, 1])
        normalized_series = mapper(series)
    else:
        normalized_series = series

    # Return as dictionary, so it's simple to append to pandas dataframe:
    datasetline = {"mean": mean(normalized_series),
                   "variance": variance(normalized_series),
                   "skewness": skew(normalized_series),
                   "kurtosis": kurtosis(normalized_series, fisher=False),
                   "Type": None}
    return datasetline


# This function plot estimated probability densities to sample data:
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