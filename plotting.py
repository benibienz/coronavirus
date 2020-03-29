import matplotlib.pyplot as plt


def timeseries(X=None, Y=None):
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(14, 8))
    ax.plot(Y)
    # ax.set_ylim((0, 100))
    plt.tight_layout()
    plt.show()


def bar(X, Y):
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(14, 8))
    ax.bar(X, Y)