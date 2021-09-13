import numpy as np
import scipy.stats as sts
import math as m

size = [10, 100, 1000]

def values(sample, average, med, zr, zq, ztr):
    sample.sort()
    average.append(np.average(sample))
    med.append(np.median(sample))
    zr.append((min(sample) + max(sample))/2)
    zq.append((np.quantile(sample, 0.25) + np.quantile(sample, 0.75))/2)
    ztr.append(np.mean(sample[len(sample)//4: len(sample)-len(sample)//4]))

def print_E_D(all_v, s):
    for i in all_v:
        E = round(np.mean(i), 6)
        D = round(np.std(i) ** 2, 6)
        print("size = ", s)
        print(E, D, round(E - D ** 0.5, 6), round(E + D ** 0.5, 6))


#Нормальное распределение
for s in size:
    average, med, zr, zq, ztr = [], [], [], [], []
    all_v = [average, med, zr, zq, ztr]
    for i in range(1000):
        sample = sts.norminvgauss.rvs(1, 0, size=s)
        values(sample, average, med, zr, zq, ztr)
    print_E_D(all_v, s)

#Распределение Коши
for s in size:
    average, med, zr, zq, ztr = [], [], [], [], []
    all_v = [average, med, zr, zq, ztr]
    for i in range(1000):
        sample = sts.cauchy.rvs(size=s)
        values(sample, average, med, zr, zq, ztr)
    print_E_D(all_v, s)

#Распределение Лапласа
for s in size:
    average, med, zr, zq, ztr = [], [], [], [], []
    all_v = [average, med, zr, zq, ztr]
    for i in range(1000):
        sample = sts.laplace.rvs(loc=0, scale=1 / m.sqrt(2), size=s)
        values(sample, average, med, zr, zq, ztr)
    print_E_D(all_v, s)

#Распределение Пуассона
for s in size:
    average, med, zr, zq, ztr = [], [], [], [], []
    all_v = [average, med, zr, zq, ztr]
    for i in range(1000):
        sample = sts.poisson.rvs(mu=10, size=s)
        values(sample, average, med, zr, zq, ztr)
    print_E_D(all_v, s)

#Равномерное распределение
for s in size:
    average, med, zr, zq, ztr = [], [], [], [], []
    all_v = [average, med, zr, zq, ztr]
    for i in range(1000):
        sample = sts.uniform.rvs(-m.sqrt(3), 2 * m.sqrt(3), size=s)
        values(sample, average, med, zr, zq, ztr)
    print_E_D(all_v, s)




