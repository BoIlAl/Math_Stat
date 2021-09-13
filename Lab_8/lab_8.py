import numpy as np
import matplotlib.pyplot as plt


def read():
    data = []
    file = open("wave_ampl.txt", 'r')
    # удаление скобок
    for l in file.readlines():
        data.append([float(element) for element in l.replace('[', '').replace(']', '').split(", ")])
    data = np.asarray(data)
    data = np.reshape(data, (data.shape[1] // 1024, 1024))

    file.close()
    return data

def draw_signal(signal):
    plt.title("Сигнал")
    plt.plot(range(len(signal)), signal, color='indigo')
    plt.show()


def draw_histogram(signal):
    bins = int(1.72 * (len(signal) ** (1 / 3)))
    hist = plt.hist(signal, bins, color='indigo')
    plt.title("Гистограмма сигнала")
    plt.show()
    return hist, bins



def get_areas(hist, bins):
    x, start, finish = [], [], []
    types = [0] * bins
    # границы столбцов гистограммы
    for i in range(bins):
        x.append(hist[0][i])
        start.append(hist[1][i])
        finish.append(hist[1][i + 1])

    sort_x = sorted(x)
    # разделение на типы
    for i in range(bins):
        if x[i] == sort_x[len(x) - 1]:
            types[i] = "фон"
        elif x[i] == sort_x[len(x) - 2]:
            types[i] = "сигнал"
        else:
            types[i] = "переход"

    return start, finish, types


def get_zones(signal, hist, bins):
    # нахождение границ гистограммы и типов
    start, finish, types = get_areas(hist, bins)

    signal_types = [0] * len(signal)
    zones, zones_type = [], []

    for i in range(len(signal)):
        for j in range(len(types)):
            if (signal[i] >= start[j]) and (signal[i] <= finish[j]):
                signal_types[i] = types[j]

    type = signal_types[0]
    start = 0

    for i in range(len(signal_types)):
        if type != signal_types[i]:
            zones_type.append(type)
            zones.append([start, i])
            start = i
            type = signal_types[i]

    if type != zones_type[len(zones_type) - 1]:
        zones_type.append(type)
        zones.append([start, len(signal) - 1])

    signal_data = []
    for borders in zones:
        part = []
        for j in range(borders[0], borders[1]):
            part.append(signal[j])
        signal_data.append(part)

    return zones, zones_type, signal_data


def draw_zones(zones, zones_types, signal_data):
    plt.title("Типы областей сигнала")
    plt.ylim([-0.5, 0])

    for i in range(len(zones)):
        if zones_types[i] == "фон":
            color = 'aqua'
        elif zones_types[i] == "сигнал":
            color = 'indigo'
        else:
            color = 'violet'
        plt.plot([element for element in range(zones[i][0], zones[i][1], 1)], signal_data[i], color=color, label=zones_types[i])
    plt.legend()
    plt.show()


def inter_var(signal):
    sum = 0.0
    mean_signal = np.empty(signal.shape[0])
    for i in range(len(signal)):
        mean_signal[i] = np.mean(signal[i])
    mean = np.mean(mean_signal)

    for i in range(len(mean_signal)):
        sum += (mean_signal[i] - mean) ** 2
    sum /= signal.shape[0]

    return len(signal) * sum


def intra_var(signal):
    result = 0.0
    for i in range(signal.shape[0]):
        mean = np.mean(signal[i])
        sum = 0.0
        for j in range(signal.shape[1]):
            sum += (signal[i][j] - mean) ** 2
        sum /= signal.shape[0]
        result += sum

    return result / signal.shape[0]


def get_k(num):
    i = 4
    while num % i != 0:
        i += 1
    return i


def get_fisher(signal, zones):
    fishers = []
    for i in range(len(zones)):
        start = zones[i][0]
        finish = zones[i][1]
        k = get_k(finish - start)

        while k == finish - start:
            finish += 1
            k = get_k(finish - start)

        data = np.reshape(signal[start:finish], (k, int(signal[start:finish].size / k)))
        fisher = inter_var(data) / intra_var(data)
        fishers.append([round(fisher, 2), k])
    return fishers


# считывание сигнала
data = read()
selectInd = 1
signal = data[selectInd]

# график сигнала
draw_signal(signal)
# гистограмма сигнала
hist, bins = draw_histogram(signal)

# разделение сигнала
zones, zones_types, signal_data = get_zones(signal, hist, bins)
# график разделенного на области сигнала
draw_zones(zones, zones_types, signal_data)

# критерий Фишера
print(get_fisher(signal, zones))
