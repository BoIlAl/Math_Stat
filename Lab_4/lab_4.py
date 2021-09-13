import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt
import math

def distrFunction(sample):
    def F(x):
        return (len(sample[sample <= x])) / len(sample)
    return F

def probDensity(size):
    distributions = (
        (stats.norm, (0, 1), 'Normal', (-4, 4), 0.05),
        (stats.cauchy, (0, 1), 'Cauchy', (-4, 4), 0.05),
        (stats.laplace, (0, 1 / math.sqrt(2)), 'Laplace', (-4, 4), 0.05),
        (stats.uniform, (-math.sqrt(3), 2 * math.sqrt(3)), 'Uniform', (-4, 4), 0.05)
    )
    d_pois = (stats.poisson, (10, 0), 'Poisson', (6, 14), 1)
    for dist in distributions:
        yield dist[0].rvs(*dist[1], size), lambda x: dist[0].pdf(x, *dist[1]), dist[2], dist[3], dist[4]
    yield d_pois[0].rvs(*d_pois[1], size), lambda x: d_pois[0].pmf(x, *d_pois[1]), d_pois[2], d_pois[3],d_pois[4]

distributions = (
        (stats.norm, (0, 1), 'Normal', (-4, 4)),
        (stats.cauchy, (0, 1), 'Cauchy', (-4, 4)),
        (stats.laplace, (0, 1 / math.sqrt(2)), 'Laplace', (-4, 4)),
        (stats.uniform, (-math.sqrt(3), 2 * math.sqrt(3)), 'Uniform', (-4, 4)),
        (stats.poisson, (10, 0), 'Poisson', (6, 14))
    )

#Эмпирическая функция распределения
for size in [20, 60, 100]:
    for num in distributions:
        sample, func, name, boarders = num[0].rvs(*num[1], size), lambda x: num[0].cdf(x, *num[1]), num[2], num[3]
        functions = (distrFunction(sample), func)
        figure, ax = plt.subplots(1, 1)
        for f in functions:
            x = np.arange(boarders[0], boarders[1], 0.05)
            ax.plot(x, list(map(f, x)))
        plt.xlim(boarders)
        plt.title(name + ' n = ' + str(size))
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.savefig(name + str(size))

#Ядерные оценки
for size in [20, 60, 100]:
    for fun in probDensity(size):
        for m in [1 / 2, 1, 2]:
            sample, dist_func, name, boarders, step = fun
            kde = stats.gaussian_kde(sample)
            kde.set_bandwidth(bw_method=kde.factor * m)
            figure, ax = plt.subplots(1, 1)

            x = np.arange(boarders[0], boarders[1], step)
            ax.plot(x, kde.pdf(x))
            x = np.arange(boarders[0], boarders[1], step)
            ax.plot(x, dist_func(x))

            plt.xlabel('x')
            plt.ylabel('f(x)')
            if m == 1/2:
                plt.title(name + ' n = ' + str(size) + ', h = h_n / 2')
            if m == 1:
                plt.title(name + ' n = ' + str(size) + ', h = h_n')
            if m == 2:
                plt.title(name + ' n = ' + str(size) + ', h = 2 h_n')
            plt.savefig(str(m) + name + str(size) + '.png')
