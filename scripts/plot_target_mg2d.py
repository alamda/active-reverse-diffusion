import numpy as np
from matplotlib import figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

sys.path.insert(0, '../src/')

from read_configs import Configs
from target_multi_gaussian_2D import TargetMultiGaussian2D

def main():
    myConfigs = Configs(fname="diffusion.conf")

    myTarget = TargetMultiGaussian2D(mu_x_list=myConfigs.mu_x_list,
                                    mu_y_list=myConfigs.mu_y_list,
                                    sigma_list=myConfigs.sigma_list,
                                    pi_list=myConfigs.pi_list,
                                    sample_size=myConfigs.sample_size,
                                    xmin=myConfigs.xmin,
                                    xmax=myConfigs.xmax,
                                    ymin=myConfigs.ymin,
                                    ymax=myConfigs.ymax,
                                    num_bins=myConfigs.num_hist_bins,
                                    target_sample_fname=myConfigs.target_fname)

    myTarget.mmap_sample()
    
    extent = [myTarget.xmin, myTarget.xmax, myTarget.ymin, myTarget.ymax]
    
    idx_arr = myTarget.target_hist_idx_arr
    prob_arr = myTarget.target_hist_idx_prob_arr

    sample_x = myTarget.sample[:,0].numpy()
    sample_y = myTarget.sample[:,1].numpy()

    fig = figure.Figure()
    fig.set_size_inches(6,2)
    fig.suptitle("Target")
    axs = fig.subplots(1,2)

    hist, _, _ = np.histogram2d(sample_x, sample_y)
    
    axs[0].imshow(prob_arr.reshape(myTarget.num_points_x, myTarget.num_points_y),
                  extent=extent)
    
    axs[0].set_title("Probability")

    axs[1].imshow(hist, extent=[myConfigs.xmin, myConfigs.xmax,
                                 myConfigs.ymin, myConfigs.ymax])
    axs[1].set_title("Sample")
    
    # divider = make_axes_locatable(axs[1])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    
    # fig.colorbar(im, cax=cax)

    fig.tight_layout()
    fig.savefig("target.png")
    
if __name__=="__main__":
    main()
