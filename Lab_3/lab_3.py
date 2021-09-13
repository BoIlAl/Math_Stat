from scipy.stats import norm, laplace, poisson, cauchy, uniform
import numpy as np
import math as m
import seaborn as sns
import matplotlib.pyplot as plt

size = [20, 100]

def drawBoxplot(tips, name):
    sns.boxplot(data=tips, orient='h', palette="Set3")
    sns.despine(offset=10)
    plt.title(name)
    plt.xlabel("x")
    plt.savefig(str(name)+'.jpg')

def len_mus(Q1, Q3):
    X1 = Q1 - 3 / 2 * (Q3 - Q1)
    X2 = Q3 + 3 / 2 * (Q3 - Q1)
    return X1, X2

def outlier(sample):
    outliers = 0
    sample.sort()
    Q1 = np.quantile(sample, 0.25)
    Q3 = np.quantile(sample, 0.75)
    X1, X2 = len_mus(Q1, Q3)
    for x in sample:
        if x < X1 or x > X2:
            outliers += 1
    return outliers

#Нормальное распределение
tips, sum_outlier = [], 0
for s in size:
    for i in range(1000):
        sample = norm.rvs(size=s)
        sum_outlier += outlier(sample)
    result = sum_outlier / (s * 1000)
    print("n =", s, "%.3f" % result)
    sample = norm.rvs(size=s)
    sample.sort()
    tips.append(sample)
drawBoxplot(tips, "Normal")

#Распределение Коши
tips, sum_outlier =  [], 0
for s in size:
    for i in range(1000):
        sample = cauchy.rvs(size=s)
        sample.sort()
        sum_outlier += outlier(sample)
    result = sum_outlier / (s * 1000)
    print("n =", s, "%.3f" % result)
    sample = cauchy.rvs(size=s)
    sample.sort()
    tips.append(sample)
drawBoxplot(tips, "Cauchy")

#Распределение Лапласа
tips, sum_outlier = [], 0
for s in size:
    for i in range(1000):
        sample = laplace.rvs(size=s, scale=1 / m.sqrt(2), loc=0)
        sample.sort()
        sum_outlier += outlier(sample)
    result = sum_outlier / (s * 1000)
    print("n =", s, "%.3f" % result)
    sample = laplace.rvs(size=s, scale=1 / m.sqrt(2), loc=0)
    sample.sort()
    tips.append(sample)
drawBoxplot(tips, "Laplace")

#Распределение Пуассона
tips, sum_outlier = [], 0
for s in size:
    for i in range(1000):
        sample = poisson.rvs(mu=10, size=s)
        sample.sort()
        sum_outlier += outlier(sample)
    result = sum_outlier / (s * 1000)
    print("n =", s, "%.3f" % result)
    sample = poisson.rvs(mu=10, size=s)
    sample.sort()
    tips.append(sample)
drawBoxplot(tips, "Poisson")

#Равномерное распределение
tips, sum_outlier = [], 0
for s in size:
    for i in range(1000):
        sample = uniform.rvs(-m.sqrt(3), 2 * m.sqrt(3), size=s)
        sample.sort()
        sum_outlier += outlier(sample)
    result = sum_outlier / (s * 1000)
    print("n =", s, "%.3f" % result)
    sample = uniform.rvs(-m.sqrt(3), 2 * m.sqrt(3), size=s)
    sample.sort()
    tips.append(sample)
drawBoxplot(tips, "Uniform")


#Теоретическая вероятность выбросов
Q1 = norm.ppf(0.25)
Q3 = norm.ppf(0.75)
X1, X2 = len_mus(Q1, Q3)
P = norm.cdf(X1) + 1 - norm.cdf(X2)
print("%.3f" % Q1, "%.3f" % Q3, "%.3f" % X1, "%.3f" % X2, "%.3f" % P)

Q1 = cauchy.ppf(0.25)
Q3 = cauchy.ppf(0.75)
X1, X2 = len_mus(Q1, Q3)
P = cauchy.cdf(X1) + 1 - cauchy.cdf(X2)
print("%.3f" % Q1, "%.3f" % Q3, "%.3f" % X1, "%.3f" % X2, "%.3f" % P)

Q1 = laplace(scale=1 / m.sqrt(2), loc=0).ppf(0.25)
Q3 = laplace(scale=1 / m.sqrt(2), loc=0).ppf(0.75)
X1, X2 = len_mus(Q1, Q3)
P = laplace(scale=1 / m.sqrt(2), loc=0).cdf(X1) + 1 - laplace(scale=1 / m.sqrt(2), loc=0).cdf(X2)
print("%.3f" % Q1, "%.3f" % Q3, "%.3f" % X1, "%.3f" % X2, "%.3f" % P)

Q1 = poisson(mu=10).ppf(0.25)
Q3 = poisson(mu=10).ppf(0.75)
X1, X2 = len_mus(Q1, Q3)
P = poisson(mu=10).cdf(X1) + 1 - poisson(mu=10).cdf(X2) - poisson(mu=10).pmf(X1)
print("%.3f" % Q1, "%.3f" % Q3, "%.3f" % X1, "%.3f" % X2, "%.3f" % P)

Q1 = uniform(-m.sqrt(3), 2 * m.sqrt(3)).ppf(0.25)
Q3 = uniform(-m.sqrt(3), 2 * m.sqrt(3)).ppf(0.75)
X1, X2 = len_mus(Q1, Q3)
P = uniform(-m.sqrt(3), 2 * m.sqrt(3)).cdf(X1) + 1 - uniform(-m.sqrt(3), 2 * m.sqrt(3)).cdf(X2)
print("%.3f" % Q1, "%.3f" % Q3, "%.3f" % X1, "%.3f" % X2, "%.3f" % P)
