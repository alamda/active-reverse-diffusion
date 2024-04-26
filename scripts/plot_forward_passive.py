import numpy as np
from matplotlib import figure
import sys

sys.path.insert(0, '../src/')

from read_configs import Configs
from target_multi_gaussian_2D import TargetMultiGaussian2D
from noise import NoisePassive
from diffusion_numeric import DiffusionNumeric

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
    
    myPassiveNoise = NoisePassive(T=myConfigs.passive_noise_T)
    
    myDiffNum = DiffusionNumeric(ofile_base=myConfigs.ofile_base,
                                 passive_noise=myPassiveNoise,
                                 target=myTarget,
                                 num_diffusion_steps=myConfigs.num_diffusion_steps,
                                 dt=myConfigs.dt,
                                 sample_size=myConfigs.sample_size)
    
    passive_forward = myDiffNum.forward_passive_data_h.mmap_tensor_from_file()
        
    extent = [myConfigs.xmin, myConfigs.xmax, myConfigs.ymin, myConfigs.ymax]
    
    time_f = 0
    time_l = myConfigs.dt * len(passive_forward)
    
    fig = figure.Figure()
    fig.set_size_inches(6,2)
    fig.suptitle("Forward Diffusion")
    
    axs = fig.subplots(1,2)
    
    img0 = axs[0].hist2d(passive_forward[0][:,0], passive_forward[0][:,1],
                         bins=myConfigs.num_hist_bins)
    
    axs[0].set_title(f't={time_f:.3f}')
    
    img1 = axs[1].hist2d(passive_forward[-1][:,0], passive_forward[-1][:,1],
                         bins=myConfigs.num_hist_bins)
    
    axs[1].set_title(f't={time_l:.3f}')
    
    fig.tight_layout()    
    fig.savefig("forward_diffusion_passive.png")

if __name__=="__main__":
    main()