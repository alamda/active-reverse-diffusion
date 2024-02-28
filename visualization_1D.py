from abstract_classes.visualization import VisualizationAbstract
import matplotlib.pyplot as plt


class Visualization1D(VisualizationAbstract):
    def __init__(self, name="viz1d", xlim=None):
        super().__init__(name=name)

    def plot_dsn(self, data=None, label=None, bins=None):
        self.ax_dsn.hist(data, density=True, alpha=0.5, label=label, bins=bins)

    def plot_diff(self, diff_arr=None, label=None):
        self.ax_diff.plot(diff_arr, label=label)
