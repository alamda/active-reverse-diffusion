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

def main():
    myConfigs = Configs(fname="diffusion.conf")
    
    myTarget = TargetMultiGaussian(mu_list=myConfigs.mu_list,
                                    sigma_list=myConfigs.sigma_list,
                                    pi_list=myConfigs.pi_list,
                                    sample_size=myConfigs.sample_size,
                                    xmin=myConfigs.xmin,
                                    xmax=myConfigs.xmax,
                                    target_sample_fname=myConfigs.target_fname)
    
if __name__=="__main__":
    main()
