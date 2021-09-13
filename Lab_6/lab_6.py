import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, linregress

# оценка коэффициентов с помощью метода наименьших квадратов(МНК)
def least_squares(x, y):
    res = linregress(x, y)
    b_1, b_0 = res.slope, res.intercept
    return b_0, b_1

# оценка коэффициентов с помощью метода наименьших модулей(МНМ)
def least_absolute(x, y):
    k_q = 1.491
    n = len(x)
    l = int(n / 4)
    j = n - l + 1
    r_Q = 0
    for i in range(n):
        r_Q += np.sign((x[i] - np.median(x)) * (y[i] - np.median(y)))
    r_Q = r_Q / n
    q_y, q_x = (y[j] - y[l]) / k_q, (x[j] - x[l]) / k_q

    b_1 = r_Q * q_y / q_x
    b_0 = np.median(y) - b_1 * np.median(x)
    return b_0, b_1

def printEst(x, y, y_model, strin):
    # подсчет с помощью метода наименьших квадратов(МНК)
    b_0, b_1 = least_squares(x, y)
    # подсчет с помощью метода наименьших модулей(МНМ)
    b_0_LAD, b_1_LAD = least_absolute(x, y)
    print("МНК ", strin, round(b_0, 2), round(b_1, 2))  # МНК
    print("МНМ ", strin, round(b_0_LAD, 2), round(b_1_LAD, 2))  # МНМ
    y_mnk = b_0 + b_1 * x
    y_mnm = b_0_LAD + b_1_LAD * x
    dist_mnk = sum([(y_model[i] - y_mnk[i]) ** 2 for i in range(len(y))])
    dist_mnm = sum([abs(y_model[i] - y_mnm[i]) for i in range(len(y))])
    print("МНК dist = " + str(dist_mnk) + ", МНМ dist = " + str(dist_mnm))

    return y_mnk, y_mnm

def Draw(x, y, y_model, y_mnk, y_mnm, strin):
    plt.plot(x, y, 'o', label='Выборка', color='darkviolet')
    plt.plot(x, y_model, label='Модель', color='royalblue', linewidth=2)
    plt.plot(x, y_mnk, label='МНК', color='violet', linewidth=2)
    plt.plot(x, y_mnm, label='МНМ', color='mediumspringgreen', linewidth=2)
    plt.legend()
    plt.title(strin)
    plt.grid()
    plt.show()

#отрезок с равномерным шагом
x = np.linspace(-1.8, 2, 20)
#нормально распределенная ошибка
e = norm.rvs(0, 1, size=20)
y_model = 2 * x + 2
y1 = 2 * x + 2 + e
y2 = 2 * x + 2 + e
y2[0], y2[19] = y2[0] + 10, y2[19] - 10
y_mnk, y_mnm = printEst(x, y1, y_model, "без возмущения")
Draw(x, y1, y_model, y_mnk, y_mnm, "Распределение без возмущения")
y_mnk, y_mnm = printEst(x, y2, y_model, "с возмущением")
Draw(x, y2, y_model, y_mnk, y_mnm, "Распределение с возмущением")

