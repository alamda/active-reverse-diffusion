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
    
    myDiffNum = DiffusionNumeric(ofile_base=myConfigs.ofile_base,
                                 passive_noise=myPassiveNoise,
                                 target=myTarget,
                                 num_diffusion_steps=myConfigs.num_diffusion_steps,
                                 dt=myConfigs.dt,
                                 sample_size=myConfigs.sample_size)

    myDiffNum.forward_diffusion_passive()
    
    myDiffNum.train_diffusion_passive(iterations=myConfigs.passive_training_iters)

if __name__=="__main__":
    main()