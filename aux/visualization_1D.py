from abstract_classes.visualization import VisualizationAbstract
import matplotlib.pyplot as plt
import torch
import scipy.special
from sklearn.neighbors import KernelDensity
import numpy as np


class Visualization1D(VisualizationAbstract):
    def __init__(self, name="viz1d", xmin=None, xmax=None,
                 bandwidth=None, kernel=None, num_points=None):
        super().__init__(name=name)

        self.set_analysis_params(xmin=xmin, xmax=xmax,
                                 bandwidth=bandwidth, kernel=kernel,
                                 num_points=num_points)

    def plot_dsn(self, data=None, label=None, bins=None):
        self.ax_dsn.hist(data, density=True, alpha=0.5, label=label, bins=bins)

    def plot_diff(self, diff_arr=None, label=None):
        self.ax_diff.plot(diff_arr[:, 0], diff_arr[:, 1], label=label)

    def set_analysis_params(self, xmin=None, xmax=None, bandwidth=None,
                            kernel=None, num_points=None):
        self.xmin = xmin
        self.xmax = xmax
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.num_points = num_points

    def approx_prob_dsn(self, data=None):
        model = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)

        if type(data) == torch.Tensor:
            model.fit(data.numpy().reshape(-1, 1))
        else:
            model.fit(data.reshape(-1, 1))

        values = np.linspace(self.xmin, self.xmax, self.num_points)
        values = values.reshape((len(values), 1))
        log_probs = model.score_samples(values)
        probs = np.exp(log_probs)
        probs = probs/np.sum(probs)

        return values, probs

    def calc_KL_div(self, data1=None, data2=None):
        _, h1 = self.approx_prob_dsn(data1)
        _, h2 = self.approx_prob_dsn(data2)

        diff = np.sum(scipy.special.rel_entr(h1, h2))

        return diff
