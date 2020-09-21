from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


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

def rmse_calculate(y_test,y_pred):
    mse  = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse )
    rmse = round(rmse, 2)
    return rmse

def plot_results(predicted_data, true_data, figtitle):
    ''' use when predicting just one analysis window '''
    fig = plt.figure(facecolor='white',figsize =(18,10))

    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.title(figtitle)
    plt.savefig('images/' + figtitle + '.png')
    plt.show()
    plt.close()
    print('Plot saved.')


def plot_results_multiple(predicted_data, true_data, prediction_len, figtitle):
    ''' use when predicting multiple analyses windows in data '''
    fig = plt.figure(facecolor='white',figsize =(18,10))
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to its correct start
    for i, data in enumerate(predicted_data):
        if i != 0:
            padding = [None for p in range(i * prediction_len)]
            plt.plot(padding + data, label='Prediction')
            plt.legend()
    plt.title(figtitle)
    # plt.savefig(figtitle + '.png')
    plt.show()
    plt.close()
    # print('Plot saved.')
        