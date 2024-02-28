from abc import ABC as AbstractBaseClass
from abc import abstractmethod
import matplotlib.pyplot as plt


class VisualizationAbstract(AbstractBaseClass):
    """Abstract class for visualizing diffusion simulations"""

    def __init__(self, name="viz"):
        self.fig_dsn, self.ax_dsn = plt.subplots()
        self.fig_diff, self.ax_diff = plt.subplots()

    def __delete__(self):
        plt.close('all')

    @abstractmethod
    def plot_target_dsn(self, data=None, label="target"):
        """Show the target data/dsn
        Uses data from self.target_data()"""

    @abstractmethod
    def plot_computed_dsn(self, data=None, label=None):
        """Add histogram of dsn sample computed by diffusion process"""

    @abstractmethod
    def plot_diff(self, diff_arr=None):
        """Show the quality of the score function
            * KL divergence for analytic starting dsns
            * Loss function for score fns learned with NNs
        """
