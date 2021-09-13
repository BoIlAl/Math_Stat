import numpy as np
from scipy.stats import laplace, uniform
import scipy.stats as stats
import math
import pandas as pd

alpha = 0.05
p = 1 - alpha

def setFreq(sample, segments):
    n_i = np.array([])
    for i in range(-1, len(segments)):
        if i == -1:
            n_i = np.append(n_i, len(sample[sample <= segments[0]]))
        elif i == len(segments) - 1:
            n_i = np.append(n_i, len(sample[sample >= segments[-1]]))
        else:
            n_i = np.append(n_i, len(sample[(sample <= segments[i + 1]) & (sample >= segments[i])]))
    return n_i

def setProbability(segments):
    p_i = np.array([])
    for i in range(-1, len(segments)):
        if i == -1:
            prev_cdf = 0
        else:
            prev_cdf = stats.norm.cdf(segments[i])
        if i == len(segments) - 1:
            cur_cdf = 1
        else:
            cur_cdf = stats.norm.cdf(segments[i + 1])
        p_i = np.append(p_i, cur_cdf - prev_cdf)
    return p_i

def fillTable(sample, segments, n, name):
    p_i = setProbability(segments)
    n_i = setFreq(sample, segments)
    sample_chi_2 = np.divide(np.multiply((n_i - n * p_i), (n_i - n * p_i)), p_i * n)
    data = []

    for i in range(0, len(n_i)):
        if i == 0:
            boarders = ['-inf', np.around(segments[0], 4)]
        elif i == len(n_i) - 1:
            boarders = [np.around(segments[-1], 4), 'inf']
        else:
            boarders = [np.around(segments[i - 1], 4), np.around(segments[i], 4)]
        data.append([boarders, n_i[i], np.around(p_i[i], 4),
                     np.around(p_i[i] * n, 4), np.around(n_i[i] - n * p_i[i], 4),
                     np.around(sample_chi_2[i], 4)])
    data.append(["-", np.sum(n_i), np.around(np.sum(p_i), 4),
                 np.around(np.sum(p_i * n), 4), np.around(np.sum(n_i - n * p_i), 4),
                 np.around(np.sum(sample_chi_2), 4)])
    pd.DataFrame(data).to_excel(name + '.xlsx', index=False)


def processParam(sample, size):
    mu = np.mean(sample)
    sigma = np.std(sample)
    k = math.ceil(1.72 * size ** (1 / 3))
    chi_2 = stats.chi2.ppf(p, k - 1)
    print('mu = ' + str(np.around(mu, 4)))
    print('sigma = ' + str(np.around(sigma, 4)))
    print('k = ' + str(k))
    print('chi_2 = ' + str(chi_2))
    segments = np.linspace(-1.1, 1.1, num=k - 1)
    return segments

def сalculation(sample, size, name):
    print(name)
    segments = processParam(sample, size)
    fillTable(sample, segments, size, name)


сalculation(np.random.normal(0, 1, size=100), 100, 'Normal')
сalculation(laplace.rvs(size=20, scale=1 / math.sqrt(2), loc=0), 20, 'Laplace')
сalculation(uniform.rvs(size=20, loc=-math.sqrt(3), scale=2 * math.sqrt(3)), 20, 'Uniform')
