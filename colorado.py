import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pd.options.display.expand_frame_repr = False


def small_testing_plot():
    df = pd.read_csv('case_data/colorado_cases.csv', index_col=0)
    df['New Cases'] = df['Cases'].diff()
    df['New Tests'] = df['Tests'].diff()
    df = df.drop(columns=['Cases', 'Tests']).iloc[1:, :]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 8))

    df.plot(kind='bar', title='Daily New Cases', ax=axes[0])
    df['% Positive'] = 100 * df['New Cases'] / df['New Tests']
    df['% Positive'].plot(kind='bar', title='Percentage of positive tests', ax=axes[1])
    axes[1].set_ylabel('%')
    fig.suptitle('Colorado cases in March', fontsize=16)
    print(df)
    plt.show()


def age_demo_study():
    dir = 'case_data/co_daily_files/'
    files = os.listdir(dir)
    files = sorted(files)[1:-11]
    # files = sorted(files)[1:]
    boulder_cases = []
    tot_cases = []
    for i, f in enumerate(files):
        date = f[27:-4]
        print(date)
        df = pd.read_csv(dir + f)

        boulder = df[df['attribute'] == 'Boulder']
        if boulder.empty:
            boulder = df[df['attribute'] == 'Boulder County']
        # print(boulder)
        boulder = boulder[boulder['metric'] == 'Cases']
        boulder_cases.append(boulder['value'].values[0])

        tot_cases.append(df.loc[0, 'value'])

        case_df = df[df['description'] == 'Case Counts by Age Group']
        if case_df.empty:
            case_df = df[df['description'] == 'COVID-19 in Colorado by Age Group']
        case_df = case_df[case_df['metric'] == 'Cases'].loc[:, ['value', 'attribute']]
        case_df = case_df.set_index('attribute')
        if i == 0:
            prev_cases = case_df['value']
            big_df = pd.DataFrame(columns=case_df.index)
        else:
            case_df['value'] -= prev_cases
            case_df[case_df < 0] = 0  # get rid of negatives
            case_df['perc'] = 100 * case_df['value'] / sum(case_df['value'])
            perc_df = case_df.drop(columns=['value']).squeeze()
            big_df.loc[date] = perc_df

    # boulder_df = pd.DataFrame(index=big_df.index)
    # boulder_df['cases'] = boulder_cases[1:]
    # print(boulder_df)
    # boulder_df.diff().plot(kind='bar')

    co_df = pd.DataFrame(index=big_df.index)
    co_df['cases'] = tot_cases[1:]
    co_df.diff()[1:].plot(kind='bar', figsize=(12, 8), width=1, edgecolor='black')
    plt.ylabel('Num cases')
    plt.title('Daily new cases in Colorado', fontdict={'size': 20})
    plt.show()

    print(big_df)
    big_df.fillna(0, inplace=True)
    colors = ['#0063E5', '#1F5BCD', '#3E54B5', '#5D4C9D', '#7C4585', '#9B3E6D', '#BA3655', '#D92F3D', '#F82825', '#999999']
    big_df[1:].plot(kind='bar', stacked=True, edgecolor='black', colors=colors, width=1, figsize=(12, 8))
    plt.xticks(range(len(big_df)), big_df.index, rotation=45)
    plt.legend(loc='right')
    plt.ylim(0, 100)
    plt.ylabel('%')
    plt.title('Age demographics of daily new cases in Colorado', fontdict={'size': 20})
    plt.show()








if __name__ == '__main__':
    age_demo_study()
