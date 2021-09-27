from numpy import log10
from numpy import mean
from numpy import var
from distribution import NormalDistribution
import scipy.stats as ss


class Hypothesis:
    def __init__(self):
        self.alpha = 0.05

    @staticmethod
    def _get_frequencies(sample, a: float, b: float, k: int):
        frequencies = [0 for _ in range(k)]
        delta = (b - a) / k
        for elem in sample:
            k_i = int((elem - a) / delta)
            frequencies[k_i] += 1
        return frequencies

    @staticmethod
    def _calculate_quantile(frequencies: [], probabilities: [], size: int):
        k = len(frequencies)
        assert k == len(probabilities)
        quantile = 0
        for i in range(k):
            quantile += ((frequencies[i] - size * probabilities[i]) ** 2) / (size * probabilities[i])
        return quantile

    def check_hypothesis(self, sample):
        size = len(sample)
        k = Hypothesis.number_of_sections(size)
        a, b = min(sample), max(sample)
        delta = (b - a) / k
        a, b = a - delta / 2, b + delta / 2
        frequencies = Hypothesis._get_frequencies(sample, a, b, k)

        distribution = NormalDistribution(mean(sample), var(sample))

        probabilities = distribution.get_probabilities(a, b, k)
        quantile = ss.chi2.ppf(1 - self.alpha / 2, k - 2)
        sample_quantile = Hypothesis._calculate_quantile(frequencies, probabilities, size)
        print(quantile)
        print(sample_quantile)
        return

    @staticmethod
    def number_of_sections(size: int):
        return int(1 + 3.3 * log10(size))
