import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
import gc
from matplotlib import figure

sys.path.insert(0, '../src/')

from read_configs import Configs
from data_proc import DataProc
from target_multi_gaussian_2D import TargetMultiGaussian2D
from noise import NoiseActive, NoisePassive
from diffusion_numeric import DiffusionNumeric
from data_handler import DiffusionSampleHandler

def main():
    myConfigs = Configs(fname="diffusion.conf")
    
    myActiveNoise = NoiseActive(Tp=myConfigs.active_noise_Tp,
                                 Ta=myConfigs.active_noise_Ta,
                                 tau=myConfigs.active_noise_tau)
    
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
                                 active_noise=myActiveNoise,
                                 target=myTarget,
                                 num_diffusion_steps=myConfigs.num_diffusion_steps,
                                 dt=myConfigs.dt,
                                 sample_size=myConfigs.sample_size,
                                 data_proc=myDataProc)

    active_models_x = myDiffNum.models_active_x_data_h.load_models()
    active_models_eta = myDiffNum.models_active_eta_data_h.load_models()
    
    time_list = [ x * myConfigs.dt for x in range(myConfigs.num_diffusion_steps, 0, -1*myConfigs.reverse_sample_step_interval)]
    time_list.reverse()
    
    time_list = np.array([10,20,30])*myConfigs.dt
    
    active_rev_time_list = []
    active_rev_diff_list = []
    
    for time in time_list:
        print(time)
        myDiffNum.sample_from_diffusion_active(all_models_x=active_models_x, all_models_eta=active_models_eta, time=time)
        myDiffNum.calculate_active_diff_list(multiproc=True)
        
        active_rev_time_list.append(time)
        active_rev_diff_list.append(myDiffNum.active_diff_list[-1])
        
    print(active_rev_time_list)
    print(active_rev_diff_list)
    
    fig = figure.Figure()
    ax = fig.subplots()
    
    ax.scatter(np.array(active_rev_time_list), np.log(np.array(active_rev_diff_list)))
    
    fig.savefig("reverse_diffusion_active.png")    

if __name__=="__main__":
    main()