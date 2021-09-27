import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def kernel(u):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-(u ** 2) / 2)


def _edenf(sample, x, hn):
    y = 0
    for elem in sample:
        y += kernel((x - elem) / hn)
    return y / (len(sample) * hn)


def edenf(sample, x, hn):
    y = []
    for elem in x:
        y.append(_edenf(sample, elem, hn))
    return y


def empirical_density_function(sample: [], color: str, label: str):
    a, b = min(sample), max(sample)
    x = np.linspace(a, b, 100)
    kde = gaussian_kde(sample, bw_method='silverman')
    y = kde.evaluate(x)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.plot(x, y, label=label, color=color)
    plt.legend()
