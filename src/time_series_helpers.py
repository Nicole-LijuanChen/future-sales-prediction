from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
import random
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
def rmse_comp(df):
    lenth = df.shape[1]
    rmse_lst =[rmse_calculate(df.actuals,df.iloc[:,i]) for i in range(lenth)]
    df_rmses = pd.DataFrame({'names':df.columns,'RMSE_or_mean':rmse_lst})
    df_rmses.iloc[0,0]='actuals_mean'
    df_rmses.iloc[0,1]=df.actuals.mean()

    df_rmses=df_rmses.sort_values(by='RMSE_or_mean', ascending=False)
    df_rmses=df_rmses.reset_index(drop=True)
    return df_rmses

    
def plot_acf_and_pacf(series, axs, lags=24*2):
    """Plot the autocorrelation and partial autocorrelation plots of a series
    on a pair of axies.
    """
    _ = plot_acf(series, ax=axs[0], lags=lags)
    _ = plot_pacf(series, ax=axs[1], lags=lags)

def plot_trend_data(ax, name, series):
    ax.plot(series.index, series.values)
    ax.set_title("Sales Trend For {}".format(name))

def plot_results(predicted_data, true_data, figtitle):
    ''' use when predicting just one analysis window '''
    fig = plt.figure(facecolor='white',figsize =(18,10))

    ax = fig.add_subplot(111)
    ax.plot(true_data, label='Actuals',color ='deepskyblue')
    plt.plot(predicted_data, label='Forecast',color='darkorange')
    plt.legend()
    plt.title(figtitle,fontsize=20)
    fig.subplots_adjust(top=0.92)
    plt.savefig('images/' + figtitle + '.png')
    plt.show()
    plt.close()
    print('Plot saved.')


def plot_results_multiple(predicted_data, true_data, labels_width, figtitle,num_cols=2):
    ''' use when predicting multiple analyses windows in data '''
    fig, axes = plt.subplots(nrows=np.ceil(labels_width/2).astype(int), ncols=num_cols,dpi=150, figsize=(15,10))
    for i, ax in enumerate( axes.flatten()):
        predicted_data.iloc[:,i].plot(legend=True, ax=ax,linestyle='--',color='darkorange').autoscale(axis='x',tight=True)
        true_data.iloc[:,i].plot(legend=True, ax=ax, label=true_data.columns[i]+'_actuals',color='deepskyblue');
        ax.set_title( f'{predicted_data.columns[i]} vs Actuals')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)

    fig.suptitle(figtitle, fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88);
    
    # plt.savefig(figtitle + '.png')
    plt.show()
    plt.close()
    # print('Plot saved.')

def plot_forecasr_multiple(df_lts,figtitle,num_rows=2):     
    fig, axes = plt.subplots(nrows=num_rows, ncols=1, dpi=150, figsize=(15,20))

    for i, ax in enumerate( axes.flatten()):
        col = random.randint(1,df_lts[3].shape[1])
        df_lts[0][df_lts[0].columns[col]].plot(legend=True, ax=ax,linestyle='--',color='darkorange',label='VAR').autoscale(axis='x',tight=True)
        df_lts[1][df_lts[1].columns[col]].plot(legend=True, ax=ax,linestyle=':',label='LSTM_10').autoscale(axis='x',tight=True)
        df_lts[2][df_lts[2].columns[col]].plot(legend=True, ax=ax,linestyle='-.',label='LSTM_500').autoscale(axis='x',tight=True)
        df_lts[3][df_lts[3].columns[col]].plot(legend=True, ax=ax,color='deepskyblue',label='Actuals');
        ax.set_title( f'{df_lts[3].columns[col]} fc vs Actuals')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=10)

    fig.suptitle(figtitle, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92);
    plt.savefig('images/3_models_Forecast_VS_Actuals.png')
    plt.show()
    plt.close()
    print('Plot saved.')
