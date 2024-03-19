from abc import ABC as AbstractBaseClass
from abc import abstractmethod

import matplotlib.pyplot as plt
import pickle


class TargetAbstract(AbstractBaseClass):
    """Abstract class for the target distribution"""

    def __init__(self, name="target", dim=None):
        self.name = name
        self.dim = dim

        self.sample = None

    @abstractmethod
    def gen_target_sample(self):
        """Define the target dsn and sample it after it was initialized"""

    def plot_target_hist(self,
                         fname="target.png",
                         title="example target sample",
                         bins=100,
                         range=None):
        fig, ax = plt.subplots()

        ax.hist(self.sample.reshape(self.dim),
                bins=bins, density=True, range=range)
        ax.set_title(title)

        plt.savefig(fname)

        plt.close(fig)
