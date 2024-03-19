from abc import ABC as AbstractBaseClass
from abc import abstractmethod

import matplotlib.pyplot as plt
import pickle
import numpy as np


class TargetAbstract(AbstractBaseClass):
    """Abstract class for the target distribution"""

    def __init__(self, name="target", dim=None, xmin=None, xmax=None):
        self.name = name
        self.dim = dim

        self.xmin = xmin
        self.xmax = xmax

        self.sample = None

        self.x_arr = None
        self.prob_arr = None

    @abstractmethod
    def gen_target_sample(self):
        """Define the target dsn and sample it after it was initialized"""

    def plot_target_hist(self,
                         fname="target.png",
                         title="example target sample",
                         bins=100,
                         hist_range=None):
        hist_range = hist_range if hist_range is not None else (
            self.xmin, self.xmax)

        fig, ax = plt.subplots()

        hist, bins, _ = ax.hist(self.sample.reshape(self.dim),
                                bins=bins, density=True, hist_range=hist_range)

        ax.set_ylim(bottom=0)

        if (self.x_arr is not None) and (self.prob_arr is not None):
            ax2 = ax.twinx()
            ax2.plot(self.x_arr, self.prob_arr, color='orange')
            ax2.set_ylim(bottom=0)

        ax.set_title(title)

        plt.savefig(fname)

        plt.close(fig)
