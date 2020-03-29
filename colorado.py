import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pd.options.display.expand_frame_repr = False

if __name__ == '__main__':
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
