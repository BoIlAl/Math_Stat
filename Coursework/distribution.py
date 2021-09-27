import scipy.special as ss
import numpy as np
from numpy.random import default_rng
from abc import ABC, abstractmethod


class Distribution(ABC):
    @abstractmethod
    def create_sample(self, size: int):
        pass

    @abstractmethod
    def cumulative(self, x):
        pass

    def get_probabilities(self, a: float, b: float, k: int):
        assert a < b
        probabilities = [0 for _ in range(k)]
        delta = (b - a) / k
        x = a + delta
        probabilities[0] = self.cumulative(x)
        for i in range(1, k - 1):
            probabilities[i] = self.cumulative(x + delta) - self.cumulative(x)
            x += delta
        probabilities[k - 1] = 1 - self.cumulative(x)
        return probabilities

    def __init__(self):
        self.rng = default_rng()


class NormalDistribution(Distribution):
    def __init__(self, loc: float, scale: float):
        super().__init__()
        self.loc = loc
        self.scale = scale

    def cumulative(self, x):
        return 1 / 2 * (1 + ss.erf((x - self.loc) / np.sqrt(2 * self.scale)))

    def create_sample(self, size: int):
        return self.rng.normal(self.loc, self.scale, size)


class LaplaceDistribution(Distribution):
    def __init__(self, loc: float, scale: float):
        super().__init__()
        self.loc = loc
        self.scale = scale

    def cumulative(self, x):
        b = self.loc
        a = self.scale
        return 1 / 2 * np.exp((x - b) / a) if x <= 0 else 1 - 1 / 2 * np.exp(-(x - b) / a)

    def create_sample(self, size: int):
        return self.rng.laplace(self.loc, self.scale, size)


class UniformDistribution(Distribution):
    def __init__(self, low: float, high: float):
        super().__init__()
        self.low = low
        self.high = high

    def cumulative(self, x):
        return (x - self.low) / (self.high - self.low)

    def create_sample(self, size: int):
        low = - 3 ** 0.5
        high = 3 ** 0.5
        return self.rng.uniform(low, high, size)