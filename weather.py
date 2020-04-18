import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from plotting import timeseries, bar

pd.options.display.expand_frame_repr = False

INCUBATION_PERIOD = 5  # days
TIME_TO_ADMISSION = 7  # days
TIME_TO_TEST = 8  # days, assuming 5 day incubation and aggressive-ish testing criteria


CASE_DF = pd.read_csv('case_data/covid_19_clean_complete.csv').drop(columns=['Lat', 'Long'])
CONF_DF = pd.read_csv('case_data/time_series_covid19_confirmed_global.csv').drop(columns=['Lat', 'Long'])
TMAX_DF = pd.read_csv('weather_data/tMax.csv').drop(columns=['Lat', 'Long'])
HUMID_DF = pd.read_csv('weather_data/humidity.csv').drop(columns=['Lat', 'Long'])


def order_regions_by_total_cases(resolution='local', n=20):
    col = 'Country/Region' if resolution == 'countries' else 'Province/State'
    col_to_drop = 'Country/Region' if resolution == 'local' else 'Province/State'
    df = CONF_DF.drop(columns=[col_to_drop])
    country_df = df.groupby(col).sum()
    regions = country_df.iloc[:, -1].sort_values(ascending=False).head(n)
    print(regions)
    return country_df, list(regions.index)


def fit_exp_func(data, exp_range=None, threshold=10, plot=True, verbose=False):
    """
    Fits exponential function to data
    Args:
        data: pd.Series
        exp_range: tuple of month strings to bound fitted function
        threshold: if no exp_range is supplied it will be bounded this threshold
         and the max value of the data
        plot: plot
        verbose: print stuff if True

    Returns: function params, error (from covariance), indices of exp region
    """

    if exp_range is None:
        lower_lim = data.index[data > threshold][0]  # idx where data is above threshold
        upper_lim = data.astype('float64').idxmax()  # idx where data is max
        exp_range = (lower_lim, upper_lim)
        if verbose:
            print(f'Exponential from {lower_lim} to {upper_lim}')

    lims = (data.index.get_loc(exp_range[0]), data.index.get_loc(exp_range[1]))
    Y = data[lims[0]:lims[1]]
    X_fit = np.array(range(len(Y)))
    f = lambda x, a, b: a * np.exp(b * x)
    params, cov = curve_fit(f, X_fit, Y.values)
    if verbose:
        print(f'Params: A = {params[0]:.1f}, B = {params[1]:.1f}')
    err = np.sqrt(np.diag(cov))
    if plot:
        exp_curve = pd.DataFrame(index=data.index, columns=['exp', '+1sd', '-1sd'])
        X_tot = np.array(range(0 - lims[0], lims[1] - lims[0] + 1))
        exp_curve.loc[:lims[1] + 1, 'exp'] = f(X_tot, *params)
        exp_curve.loc[:lims[1] + 1, '+1sd'] = f(X_tot, params[0], params[1] + err[1])
        exp_curve.loc[:lims[1] + 1, '-1sd'] = f(X_tot, params[0], params[1] - err[1])
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(14, 8))
        data.plot(style='x', ax=ax)
        exp_curve['exp'].plot(ax=ax, colors='g')
        exp_curve.loc[:, ['+1sd', '-1sd']].plot(ax=ax, color=['r', 'r'])
        ax.set_title(f'Daily new cases for {data.name}')
        fig.tight_layout()
        plt.show()

    return params, err, exp_range


def exp_study(resolution='local'):
    # test_thresholds = np.arange(10, 110, 10)
    test_thresholds = np.arange(5, 21, 1)
    # test_thresholds = [10]
    df, regions = order_regions_by_total_cases(resolution=resolution, n=50)
    if resolution == 'countries':
        regions.remove('China')  # exponential part of pandemic was earlier than these records go
        regions.remove('Iran')  # extremely weird
        regions.remove('Norway')  # single testing outlier and lack of overall cases makes exp function fail
        regions.remove('Qatar')  # crazy outlier
        regions.remove('Diamond Princess')  # not a region
        regions.remove('Estonia')  # negative fit
        regions.remove('Slovenia')  # negative fit
        print(df.head())
    else:
        print(df)
    # raise

    exp_coeffs, errs, exp_range_low, exp_range_high = {}, {}, {}, {}
    B_df = pd.DataFrame(index=regions, columns=test_thresholds)
    for thresh in test_thresholds:
        for region in regions:
            daily_new_cases = df[df.index == region].squeeze().diff()
            # print(daily_new_cases)
            params, err, exp_range = fit_exp_func(daily_new_cases, threshold=thresh, plot=False)
            exp_coeffs[region] = params[1]
            errs[region] = err[1]
            exp_range_low[region], exp_range_high[region] = exp_range

        params_df = pd.DataFrame({'B': exp_coeffs, 'Error': errs, 'Exp range low': exp_range_low,
                                  'Exp range high': exp_range_high})

        # print(params_df)
        for c, row in params_df.iterrows():
            B_df.loc[c, thresh] = row['B']

    B_df['avg'] = B_df.median(axis=1)
    B_df['std'] = B_df.std(axis=1)
    sorted_by_std = B_df.sort_values(by='std')
    std_thresh = 0.05  # choose a reasonable number
    B_df = sorted_by_std[sorted_by_std['std'] < std_thresh]
    B_df = B_df.sort_values(by='avg')
    print(B_df)

    # get the threshold value that produced the median B
    best_low_thresh_dict = {}
    for reg, row in B_df.iterrows():
        best_low_thresh = row.where(row == row['avg']).first_valid_index()
        if best_low_thresh == 'avg':
            best_low_thresh = 10  # just set it to 10
        best_low_thresh_dict[reg] = best_low_thresh

    # loop again with best low thresholds
    exp_coeffs, errs, exp_range_low, exp_range_high = {}, {}, {}, {}
    for region in best_low_thresh_dict.keys():
        daily_new_cases = df[df.index == region].squeeze().diff()
        params, err, exp_range = fit_exp_func(daily_new_cases,
                                              threshold=best_low_thresh_dict[region], plot=False)
        exp_coeffs[region] = params[1]
        errs[region] = err[1]
        exp_range_low[region], exp_range_high[region] = exp_range

    best_params_df = pd.DataFrame(
        {'B': exp_coeffs, 'Error': errs, 'Exp range low': exp_range_low,
         'Exp range high': exp_range_high})

    # fit_exp_func(df[df.index == 'Finland'].squeeze().diff(), threshold=10)

    return best_params_df


def temp_study(resolution='local'):
    exp_params_df = exp_study(resolution=resolution)
    avg_tmax, avg_humidity = [], []
    for region, row in exp_params_df.iterrows():
        col = 'Country/Region' if resolution == 'countries' else 'Preqer'
        tmax = TMAX_DF.loc[TMAX_DF[col] == region].squeeze()
        humidity = HUMID_DF.loc[HUMID_DF[col] == region].squeeze()
        if type(tmax) == pd.DataFrame:
            tmax = tmax.mean(axis=0)  # average over local
        if type(humidity) == pd.DataFrame:
            humidity = humidity.mean(axis=0)  # average over local

        # lag weather by onset time to test
        low_i = tmax.index.get_loc(row['Exp range low']) - TIME_TO_TEST
        high_i = tmax.index.get_loc(row['Exp range high']) - TIME_TO_TEST
        avg_tmax.append(tmax.iloc[low_i:high_i].mean())
        avg_humidity.append(humidity.iloc[low_i:high_i].mean())

    exp_params_df['Avg tMax'] = avg_tmax
    exp_params_df['Avg humidity'] = avg_humidity
    print(exp_params_df.sort_values(by='B'))
    sns.lmplot(x='Avg tMax', y='B', data=exp_params_df, fit_reg=True)
    sns.lmplot(x='Avg humidity', y='B', data=exp_params_df, fit_reg=True)
    plt.show()


if __name__ == '__main__':
    temp_study(resolution='countries')
