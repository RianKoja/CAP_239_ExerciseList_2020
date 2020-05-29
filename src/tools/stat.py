########################################################################################################################
# Used to compute statisticas parameters of data, used in several exercises, such as skewness and kurtosis.
#
# Written by Rian Koja to publish in a GitHub repository with specified licence.
########################################################################################################################


# Standard imports:
from statistics import mean, variance
from scipy.stats import moment, skew, kurtosis
from scipy.interpolate import interp1d


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
