import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from plotting import timeseries, bar

pd.options.display.expand_frame_repr = False

INCUBATION_PERIOD = 5  # days
TIME_TO_ADMISSION = 7  # days
TIME_TO_TEST = 8  # days, assuming 5 day incubation and aggressive-ish testing criteria


CASE_DF = pd.read_csv('case_data/covid_19_clean_complete.csv').drop(columns=['Lat', 'Long'])
CONF_DF = pd.read_csv('case_data/time_series_covid19_confirmed_global.csv').drop(columns=['Lat', 'Long'])
TMAX_DF = pd.read_csv('weather_data/tMax.csv').drop(columns=['Lat', 'Long'])


def total_cases():
    print(CONF_DF.head())
    

def fit_exp_func(data, exp_range=None, threshold=10, plot=True):
    """
    Fits exponential function to data
    Args:
        data: pd.Series
        exp_range: tuple of month strings to bound fitted function
        threshold: if no exp_range is supplied it will be bounded this threshold
         and the max value of the data
        plot: plot

    Returns: function params
    """

    if exp_range is None:
        lower_lim = data.index[data > threshold][0]  # idx where data is above threshold
        upper_lim = data.astype('float64').idxmax()  # idx where data is max
        exp_range = (lower_lim, upper_lim)
        print(f'Exponential from {lower_lim} to {upper_lim}')

    lims = (data.index.get_loc(exp_range[0]), data.index.get_loc(exp_range[1]))
    Y = data[lims[0]:lims[1]]
    X_fit = np.array(range(len(Y)))
    f = lambda x, a, b: a * np.exp(b * x)
    params, cov = curve_fit(f, X_fit, Y.values)
    print(f'Params: A = {params[0]}, B = {params[1]}')

    if plot:
        exp_curve = data.copy()
        exp_curve[:] = np.nan
        X_tot = np.array(range(0 - lims[0], lims[1] - lims[0] + 1))
        exp_curve[:lims[1] + 1] = f(X_tot, *params)
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(14, 8))
        data.plot(style='x', ax=ax)
        exp_curve.plot(ax=ax)
        fig.tight_layout()
        plt.show()
    return params


def temp_study():
    pass


if __name__ == '__main__':
    exp_range = ('2/19/20', '3/3/20')
    # df = CONF_DF.loc[CONF_DF['Country/Region'] == 'Korea, South'].squeeze()[2:].diff()
    df = CONF_DF.loc[CONF_DF['Country/Region'] == 'Germany'].squeeze()[2:].diff()
    print(df.tail())
    fit_exp_func(df)

    tmax = TMAX_DF.loc[TMAX_DF['Country/Region'] == 'Korea, South'].squeeze()