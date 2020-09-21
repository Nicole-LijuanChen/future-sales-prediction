from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np


def reindex_to_data_frame(summary_series, df, freq):
    """Reindex a series of summary statistics to match the datetime index of
    a data frame.

    Parameters
    ----------
    summary_series: A pandas.Series of summary statistics created from a
        column in a dataframe.  For example, these could be monthly or annual
        means.

    df: A DataFrame.  The one used to construct summary_series.

    freq: A string frequency indicator.  Should match the frequency of the
        index to df.

    Returns
    -------
    reindexed_series: A Series containing the same information as
        summary_series, but reindexed to match the index of the data frame.

    Notes:
        NAs in the reindexing process are forward filled, so make sure that
    when the index of df represents date *intervals* (i.e. a monthly index)
    the left hand index is used.
    """
    min_date = df.index.min()
    resampled = summary_series.resample(freq).ffill()[min_date:]
    return resampled.reindex(df.index).ffill()


def to_col_vector(arr):
    """Convert a 1-dim numpy array into a column vector (i.e. an array with
    shape (n, 1).
    """
    return arr.reshape(-1, 1)


def plot_acf_and_pacf(series, axs, lags=24*2):
    """Plot the autocorrelation and partial autocorrelation plots of a series
    on a pair of axies.
    """
    _ = plot_acf(series, ax=axs[0], lags=lags)
    _ = plot_pacf(series, ax=axs[1], lags=lags)

