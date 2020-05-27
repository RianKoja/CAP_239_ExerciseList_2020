

import statistics
from scipy.stats import moment
from scipy.interpolate import interp1d


# This functions takes a series of data, linearly interpolates it to the [0, 1] interval,
# Then computes its sample statistical moments, returns it in dictionary form to be appended in a data frame.
def series2datasetline(series, normalize):
    if normalize:  # Normalize to the [0, 1] interval:
        mapper = interp1d([min(series), max(series)], [0, 1])
        normalized_series = mapper(series)
    else:
        normalized_series = series

    # Cannot use moment 1 as it would always be zero. Also not used anyway.
    datasetline = {"mean": statistics.mean(normalized_series),
                   "variance": moment(normalized_series, moment=2),
                   "skewness": moment(normalized_series, moment=3),
                   "kurtosis": moment(normalized_series, moment=4),
                   "Type": None}
    return datasetline
