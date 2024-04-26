import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
import gc

sys.path.insert(0, '../src/')

from read_configs import Configs
from data_proc import DataProc
from target_multi_gaussian_2D import TargetMultiGaussian2D
from noise import NoiseActive, NoisePassive
from diffusion_numeric import DiffusionNumeric
from data_handler import DiffusionSampleHandler

from matplotlib import figure

def main():
    myConfigs = Configs(fname="diffusion.conf")
    
    myPassiveNoise = NoisePassive(T=myConfigs.passive_noise_T)
    
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
    
    myDataProc = DataProc(xmin=myConfigs.xmin,
                          xmax=myConfigs.xmax,
                          ymin=myConfigs.ymin,
                          ymax=myConfigs.ymax,
                          num_hist_bins=(myConfigs.num_hist_bins, myConfigs.num_hist_bins))
    
    myDiffNum = DiffusionNumeric(ofile_base=myConfigs.ofile_base,
                                 passive_noise=myPassiveNoise,
                                 target=myTarget,
                                 num_diffusion_steps=myConfigs.num_diffusion_steps,
                                 dt=myConfigs.dt,
                                 sample_size=myConfigs.sample_size,
                                 data_proc=myDataProc)

    passive_models = myDiffNum.models_passive_data_h.load_models()
    
    time_list = [ x * myConfigs.dt for x in range(myConfigs.num_diffusion_steps, 0, -1*myConfigs.reverse_sample_step_interval)]
    time_list.reverse()
    
    passive_rev_time_list = []
    passive_rev_diff_list = []
    
    for time in time_list:
        print(time)
        myDiffNum.sample_from_diffusion_passive(all_models=passive_models, time=time)
        myDiffNum.calculate_passive_diff_list(multiproc=True)
                
        passive_rev_time_list.append(time)
        passive_rev_diff_list.append(myDiffNum.passive_diff_list[-1])
        
    print(passive_rev_time_list)
    print(passive_rev_diff_list)
    
    fig = figure.Figure()
    ax = fig.subplots()
    
    ax.scatter(np.array(passive_rev_time_list), np.log(np.array(passive_rev_diff_list)))
    
    fig.savefig("reverse_diffusion_passive.png")

if __name__=="__main__":
    main()