import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
import gc

sys.path.insert(0, '../src/')

from read_configs import Configs
from data_proc import DataProc
from target_multi_gaussian_1D import TargetMultiGaussian
from noise import NoiseActive, NoisePassive
from diffusion_numeric import DiffusionNumeric
from data_handler import DiffusionSampleHandler

def plot_loss_history(loss_arr=None, fname="loss.png"):
    plt.plot(loss_arr)
    plt.ylim((0,1))
    plt.savefig(fname)
    plt.close()

def main():
    myConfigs = Configs(fname="diffusion.conf")
    
    myActiveNoise = NoiseActive(Tp=myConfigs.active_noise_Tp,
                                 Ta=myConfigs.active_noise_Ta,
                                 tau=myConfigs.active_noise_tau)
    
    myTarget = TargetMultiGaussian(mu_list=myConfigs.mu_list,
                                   sigma_list=myConfigs.sigma_list,
                                    pi_list=myConfigs.pi_list,
                                    sample_size=myConfigs.sample_size,
                                    xmin=myConfigs.xmin,
                                    xmax=myConfigs.xmax,
                                    num_bins=myConfigs.num_hist_bins,
                                    target_sample_fname=myConfigs.target_fname)
    
    myDiffNum = DiffusionNumeric(ofile_base=myConfigs.ofile_base,
                                 active_noise=myActiveNoise,
                                 target=myTarget,
                                 num_diffusion_steps=myConfigs.num_diffusion_steps,
                                 dt=myConfigs.dt,
                                 sample_size=myConfigs.sample_size)

    myDiffNum.forward_diffusion_active()
    
    myDiffNum.train_diffusion_active(iterations=myConfigs.active_training_iters)

    plot_loss_history(loss_arr=myDiffNum.loss_history_active_x)

if __name__=="__main__":
    main()