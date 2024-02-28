from abc import ABC as AbstractBaseClass
from abc import abstractmethod
import matplotlib.pyplot as plt


class VisualizationAbstract(AbstractBaseClass):
    """Abstract class for visualizing diffusion simulations"""

    def __init__(self, name="viz"):
        self.name = name
        self.fig_dsn, self.ax_dsn = plt.subplots()
        self.fig_diff, self.ax_diff = plt.subplots()

    def __delete__(self):
        plt.close('all')

    @abstractmethod
    def plot_dsn(self, data=None, label=None, bins=None):
        """Add histogram of dataset to dsn figure"""

    @abstractmethod
    def plot_diff(self, diff_arr=None, label=None):
        """Show the quality of the score function
            * KL divergence for analytic starting dsns
            * Loss function for score fns learned with NNs
        """

    def show_plots(self):
        plt.show()

    @abstractmethod
    def set_analysis_params(self):
        """Set params needed to calculate the diff between target and computed data"""

    @abstractmethod
    def calc_KL_div(self, data1=None, data2=None):
        """Calculate KL divergence between two datasets"""

    def calc_diff(self, target_data=None, calc_data=None):
        diff_list = []

        num_steps = len(calc_data)

        for t_idx in range(0, num_steps - 1):
            data1 = target_data
            data2 = calc_data[t_idx][1]

            diff = self.calc_KL_div(data1=data1, data2=data2)

            diff_list.append([t_idx, diff])

        return diff_list
