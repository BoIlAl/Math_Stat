import pandas as pd
from numpy import mean


class Analyzer:
    def __init__(self, filename: str):
        self.data = self._read_file(filename)
        self.names = ['Sex', 'Length',	'Diam',	'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings']
        self.pd_data = pd.read_csv(filename, sep=',')

    def get_data(self):
        return self.data

    def get_axis_data(self, ind: int):
        assert 0 <= ind < len(self.data[0])
        axis_data = []
        for line in self.data:
            axis_data.append(line[ind])
        return axis_data

    def get_axis_name(self, ind: int):
        assert 0 <= ind < len(self.data[0])
        return self.names[ind]

    def get_one(self, ind: int):
        sample = []
        for x in self.data:
            if x[0] == ind:
                sample.append(x)
        return sample

    @staticmethod
    def _read_file(filename: str):
        fin = open(filename, 'r')
        lines = fin.readlines()
        converted_data = [line.split(',') for line in lines]
        for i in range(len(lines)):
            converted_data[i][0] = Analyzer.get_id(converted_data[i][0])

        float_data = []
        for line in converted_data:
            float_data.append([float(elem) for elem in line])
        return float_data

    @staticmethod
    def get_id(name: str):
        if name == 'M':
            return 0
        if name == 'F':
            return 1
        return 2

    @staticmethod
    def bins_num(size: int):
        return int(1.72 * size ** (1 / 3))

    @staticmethod
    def pearson_correlation(x, y):
        size = len(x)
        assert size == len(y)

        x_mean = mean(x)
        y_mean = mean(y)
        num = 0
        s1, s2 = 0, 0
        for i in range(0, size):
            delta_x = x[i] - x_mean
            delta_y = y[i] - y_mean
            num += delta_x * delta_y
            s1 += delta_x * delta_x
            s2 += delta_y * delta_y
        norm_cof = 1 / size
        return norm_cof * num / (norm_cof * s1 * norm_cof * s2)**(1 / 2)
