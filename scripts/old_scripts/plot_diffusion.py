import sys

sys.path.insert(0, '../src/')

from plotter_1D import Plotter
from read_configs_1D import Configs

import os
import pickle

if __name__ == "__main__":
    myConfigs = Configs(filename='diffusion.conf')

    if os.path.isfile(f"{myConfigs.ofile_base}.pkl"):
        with open(f"{myConfigs.ofile_base}.pkl", 'rb') as f:
            myDiffNum = pickle.load(f)

        myPlotter = Plotter(diffusion_object=myDiffNum)

        # myPlotter.plot_loss(ymax=1)

        myPlotter.plot_sample_hist_pre_diffusion()
        myPlotter.plot_sample_hist_post_diffusion()

        myPlotter.plot_KL_diffusion()

        myPlotter.plot_hist_animation(ymax=1.2)
