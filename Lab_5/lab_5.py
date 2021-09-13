import scipy.stats as stats
import numpy as np
import math as m
from math import pi, cos, sin
from matplotlib import pyplot as plt

def r_Q(sample):
    n1 = n2 = n3 = n4 = 0
    med_x = 0
    med_y = 0
    for s in sample:
        if s[0] > med_x and s[1] > med_y:
            n1 += 1
        if s[0] < med_x and s[1] > med_y:
            n2 += 1
        if s[0] < med_x and s[1] < med_y:
            n3 += 1
        if s[0] > med_x and s[1] < med_y:
            n4 += 1
    return ((n1 + n3) - (n2 + n4))/len(sample)

def PrintAll(r_all, r_s_all, r_q_all):
    print('r ' + 'E(z) = ', round(np.mean(r_all), 3))
    print('r_S ' + 'E(z) = ', round(np.mean(r_s_all), 3))
    print('r_Q ' + 'E(z) = ', round(np.mean(r_q_all), 3))

    print('r ' + 'E(z^2) = ', round(np.mean([r_all[i] ** 2 for i in range(len(r_all))]), 3))
    print('r_S ' + 'E(z^2) = ', round(np.mean([r_s_all[i] ** 2 for i in range(len(r_s_all))]), 3))
    print('r_Q ' + 'E(z^2) = ', round(np.mean([r_q_all[i] ** 2 for i in range(len(r_q_all))]), 3))

    print('r ' + 'D(z) = ', round(np.var(r_all), 3))
    print('r_S ' + 'D(z) = ', round(np.var(r_s_all), 3))
    print('r_Q ' + 'D(z) = ', round(np.var(r_q_all), 3))

def DrawEllipse(size, ro):
    sig_x = 1
    sig_y = 1
    # координаты центра
    u = 0
    v = 0
    #
    a = sig_x
    b = sig_y
    if sig_x ** 2 - sig_y ** 2 == 0:
        t_rot = pi / 4;
    else:
        t_rot = 0.5 * m.atan(2 * ro * sig_y * sig_x / (sig_x ** 2 - sig_y ** 2))

    a = sig_x ** 2 * m.cos(t_rot) ** 2 + ro * sig_x * sig_y * 1 + sig_y ** 2 * m.sin(t_rot) ** 2
    b = sig_x ** 2 * m.sin(t_rot) ** 2 - ro * sig_x * sig_y * 1 + sig_y ** 2 * m.cos(t_rot) ** 2
    sample = stats.multivariate_normal(mean=[0, 0], cov=[[sig_x, ro], [ro, sig_y]]).rvs(size=size)

    constant = 3
    a = a ** 0.5 * constant
    b = b ** 0.5 * constant

    t = np.linspace(0, 2 * pi, 100)
    Ellipse = np.array([a * np.cos(t), b * np.sin(t)])
    R_rot = np.array([[cos(t_rot), -sin(t_rot)], [sin(t_rot), cos(t_rot)]])
    Ellipse_rot = np.zeros((2, Ellipse.shape[1]))
    for i in range(Ellipse.shape[1]):
        Ellipse_rot[:, i] = np.dot(R_rot, Ellipse[:, i])

    plt.plot(u + Ellipse_rot[0, :], v + Ellipse_rot[1, :])
    plt.plot(sample.transpose()[0], sample.transpose()[1], 'o')
    plt.title("n = " + str(size) + "   ro = " + str(ro))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

#двумерные нормальные распределения
for size in [20, 60, 100]:
    print('size = ', size)
    for ro in [0, 0.5, 0.9]:
        print('ro = ', ro)
        r_s_all = []
        r_q_all = []
        r_all = []
        for i in range(1000):
            sample = stats.multivariate_normal(mean=[0, 0], cov=[[1, ro], [ro, 1]]).rvs(size=size)
            # коэффиценты корреляции
            r_all.append(stats.pearsonr(np.transpose(np.transpose(sample)[0]), np.transpose(sample)[1])[0])
            r_s_all.append(stats.spearmanr(sample)[0])
            r_q_all.append(r_Q(sample))
        PrintAll(r_all, r_s_all, r_q_all)
        DrawEllipse(size, ro)

#смесь нормальных распределений
for size in [20, 60, 100]:
    print('size = ', size)
    r_s_all = []
    r_q_all = []
    r_all = []
    for i in range(1000):
        sample_1 = stats.multivariate_normal(mean=[0, 0], cov=[[1, 0.9], [0.9, 1]]).rvs(size=size)
        sample_2 = stats.multivariate_normal(mean=[0, 0], cov=[[10, -0.9], [-0.9, 10]]).rvs(size=size)
        sample = 0.9 * sample_1 + 0.1 * sample_2
        #коэффиценты корреляции
        r_all.append(stats.pearsonr(np.transpose(np.transpose(sample)[0]), np.transpose(sample)[1])[0])
        r_s_all.append(stats.spearmanr(sample)[0])
        r_q_all.append(r_Q(sample))
    PrintAll(r_all, r_s_all, r_q_all)
