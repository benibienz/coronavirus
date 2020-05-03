import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
from plotting import timeseries, bar

pd.options.display.expand_frame_repr = False

INCUBATION_PERIOD = 5  # days
TIME_TO_ADMISSION = 7  # days
TIME_TO_TEST = 8  # days, assuming 5 day incubation and aggressive-ish testing criteria


CASE_DF = pd.read_csv('case_data/covid_19_clean_complete.csv').drop(columns=['Lat', 'Long'])
CONF_DF = pd.read_csv('case_data/time_series_covid19_confirmed_global.csv').drop(columns=['Lat', 'Long'])
CONF_DF_NO_CHINA = CONF_DF[CONF_DF['Country/Region'] != 'China']
TMAX_DF = pd.read_csv('weather_data/tMax.csv').drop(columns=['Lat', 'Long'])
HUMID_DF = pd.read_csv('weather_data/humidity.csv').drop(columns=['Lat', 'Long'])


def order_regions_by_total_cases(resolution='local', n=20):
    col = 'Country/Region' if resolution == 'countries' else 'Province/State'
    col_to_drop = 'Country/Region' if resolution == 'local' else 'Province/State'
    df = CONF_DF_NO_CHINA.drop(columns=[col_to_drop])
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
        plot: plot or pass in ax
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
        if isinstance(plot, bool):
            fig, ax = plt.subplots(1, 1, sharex=True, figsize=(14, 8))
        else:
            ax = plot
        data.plot(style='x', ax=ax)
        exp_curve['exp'].plot(ax=ax, colors='g')
        exp_curve.loc[:, ['+1sd', '-1sd']].plot(ax=ax, color=['r', 'r'])
        ax.set_title(f'Daily new cases for {data.name}')
        if isinstance(plot, bool):
            fig.tight_layout()
            plt.show()

    return params, err, exp_range


def exp_study(resolution='local', n=30):
    # test_thresholds = np.arange(10, 110, 10)
    test_thresholds = np.arange(5, 21, 1)
    # test_thresholds = [10]
    df, regions = order_regions_by_total_cases(resolution=resolution, n=n)
    if resolution == 'countries':
        regions.remove('China')  # exponential part of pandemic was earlier than these records go
        regions.remove('Iran')  # extremely weird
        # regions.remove('Norway')  # single testing outlier and lack of overall cases makes exp function fail
        # regions.remove('Qatar')  # crazy outlier
        # regions.remove('Estonia')  # negative fit
        # regions.remove('Slovenia')  # negative fit
        # regions.remove('Uruguay')  # negative fit
        # print(df.head())
    else:
        pass
        # regions.remove('Grand Princess')  # not a region
    # regions.remove('Diamond Princess')  # not a region

    exp_coeffs, errs, exp_range_low, exp_range_high = {}, {}, {}, {}
    B_df = pd.DataFrame(index=regions, columns=test_thresholds)
    for thresh in test_thresholds:
        for region in regions:
            daily_new_cases = df[df.index == region].squeeze().diff()
            # print(daily_new_cases)
            try:
                params, err, exp_range = fit_exp_func(daily_new_cases, threshold=thresh, plot=False)
            except:
                continue  # can't fit exp function
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
    # print(B_df)

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


def weather_study(resolution='countries', features=None, n=30):
    exp_params_df = exp_study(resolution=resolution, n=n)
    weather_features = ['tMax', 'humidity'] if features is None else features
    avg_weather_features = {wf: [] for wf in weather_features}
    for region, row in exp_params_df.iterrows():
        col = 'Country/Region' if resolution == 'countries' else 'Province/State'
        for wf in weather_features:
            df = pd.read_csv(f'weather_data/{wf}.csv').drop(columns=['Lat', 'Long'])
            data = df.loc[df[col] == region].squeeze()
            if type(data) == pd.DataFrame:
                data = data.median(axis=0)  # average over local

            # lag weather by onset time to test
            low_i = data.index.get_loc(row['Exp range low']) - TIME_TO_TEST
            high_i = data.index.get_loc(row['Exp range high']) - TIME_TO_TEST
            avg_weather_features[wf].append(data.iloc[low_i:high_i].mean())

    fig, axes = plt.subplots(1, len(weather_features), sharey=True,
                             figsize=(6 * len(weather_features), 6))
    for i, wf in enumerate(weather_features):
        display = exp_params_df.copy()
        display[wf] = avg_weather_features[wf]
        print(display.sort_values(by=wf))
        m, c, r, p, std_err = stats.linregress(display[wf], display['B'])
        sns.regplot(x=wf, y='B', data=display, fit_reg=True,
                    line_kws={'label': f'R Value: {r:.2f}'}, ax=axes[i])
        axes[i].set_ylim(bottom=0)
        axes[i].legend()
    fig.suptitle(
        f'The effect of weather on exponential coefficient of daily new cases (top {n} countries)')
    # fig.tight_layout()
    plt.show()


def weather_study_within_regions(features=None, n=1):
    fig, axes = plt.subplots(2, 1, sharex=False, figsize=(10, 8))
    df, regions = order_regions_by_total_cases(resolution='countries', n=1)
    # print(df)
    country = 'Italy'
    daily_new_cases = df[df.index == country].squeeze().diff()
    params, err, exp_range = fit_exp_func(daily_new_cases, threshold=8, plot=False)
    print(exp_range)

    low_i = daily_new_cases.index.get_loc(exp_range[0])
    high_i = daily_new_cases.index.get_loc(exp_range[1])
    pct_change = daily_new_cases.pct_change()
    normalized_pct_changes = pct_change.iloc[low_i:high_i].reset_index(drop=True)

    wf = 'tMax'
    weather_df = pd.read_csv(f'weather_data/{wf}.csv').drop(columns=['Lat', 'Long'])
    tmax = weather_df[weather_df['Country/Region'] == country].squeeze()
    low_i = tmax.index.get_loc(exp_range[0]) - TIME_TO_TEST
    high_i = tmax.index.get_loc(exp_range[1]) - TIME_TO_TEST
    normalized_tmax = tmax.iloc[low_i:high_i].reset_index(drop=True).astype(float)

    normalized_pct_changes.plot(ax=axes[0])
    axes[0].lines[-1].set_label(f'tMax {TIME_TO_TEST} days before')
    axes[0].set_title(f'% change in daily cases and tMax {TIME_TO_TEST} days before for {country}')
    ax2 = axes[0].twinx()
    normalized_tmax.plot(ax=ax2, color='m')
    axes[0].set_ylabel('%')
    ax2.lines[-1].set_label('Daily change in new cases')
    ax2.set_ylabel('C')
    axes[0].set_xlabel('Days')
    axes[0].legend(loc='upper left')
    ax2.legend(loc='upper right')

    # print(type(normalized_tmax))
    # print(type(normalized_pct_changes))

    combined = pd.concat([normalized_tmax, normalized_pct_changes], axis=1)
    combined = combined.replace([np.inf, np.nan], 1)
    # print(combined.iloc[:, 0])
    m, c, r, p, std_err = stats.linregress(combined.iloc[:, 0], combined.iloc[:, 1])
    print(combined)
    sns.scatterplot(x=combined.columns[0], y=combined.columns[1], data=combined, ax=axes[1])
    x = combined.iloc[:, 0].values
    print(m, c, r)
    axes[1].plot(x, m*x + c, 'r', label=f'R Value: {r:.2f}')
    axes[1].legend()
    axes[1].set_title(f'{wf} vs % change in daily new cases')
    axes[1].set_ylabel('Daily change in new cases (%)')
    axes[1].set_xlabel(f'tMax {TIME_TO_TEST} days before (C)')
    fig.tight_layout()
    plt.show()






if __name__ == '__main__':
    all_weather_features = ['cloud', 'dew', 'ozone', 'precip', 'pressure', 'tMax', 'tMin', 'uv',
                            'wind', 'humidity']
    features = ['tMax', 'humidity']
    # weather_study(n=30, features=features)
    weather_study_within_regions()

