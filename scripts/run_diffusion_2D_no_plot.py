import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys

sys.path.insert(0, '../src/')

from data_proc import DataProc
from target_multi_gaussian_2D import TargetMultiGaussian2D
from noise import NoiseActive, NoisePassive
from diffusion_numeric import DiffusionNumeric
from memory_profiler import profile

import matplotlib
matplotlib.use('Agg')

@profile
def main():
    ofile_base = "data"

    sample_size = 10000

    passive_noise_T = 1.0

    active_noise_Tp = 0.0
    active_noise_Ta = 1.0
    active_noise_tau = 0.25

    mu_x_list = [-2, 2]
    mu_y_list = [0, 0]
    sigma_list = [1, 1]
    pi_list = [1, 1]

    xmin = -2
    xmax = 2
    ymin = -1
    ymax = 1

    num_diffusion_steps = 100
    dt = 0.005
    
    num_hist_bins = 10
    
    num_passive_iterations=10
    num_active_iterations=50

    myPassiveNoise = NoisePassive(T=passive_noise_T,
                                  dim=sample_size)

    myActiveNoise = NoiseActive(Tp=active_noise_Tp,
                                Ta=active_noise_Ta,
                                tau=active_noise_tau,
                                dim=sample_size)

    myTarget = TargetMultiGaussian2D(mu_x_list=mu_x_list,
                                     mu_y_list=mu_y_list,
                                     sigma_list=sigma_list,
                                     pi_list=pi_list,
                                     sample_size=sample_size,
                                     xmin=xmin, xmax=xmax,
                                     ymin=ymin, ymax=ymax,
                                     num_bins=num_hist_bins)

    myDataProc = DataProc(xmin=xmin, xmax=xmax, 
                          ymin=ymin, ymax=ymax, 
                          num_hist_bins=num_hist_bins)

    myDiffNum = DiffusionNumeric(ofile_base=ofile_base,
                                 passive_noise=myPassiveNoise,
                                 active_noise=myActiveNoise,
                                 target=myTarget,
                                 num_diffusion_steps=num_diffusion_steps,
                                 dt=dt,
                                 sample_size=sample_size,
                                 data_proc=myDataProc)

    myDiffNum.forward_diffusion_passive()
    myDiffNum.forward_diffusion_active()
    
    myDiffNum.train_diffusion_passive(iterations=num_passive_iterations)

    myDiffNum.sample_from_diffusion_passive()
    myDiffNum.calculate_passive_diff_list()
    
    myDiffNum.train_diffusion_active(iterations=num_active_iterations)
    
    myDiffNum.sample_from_diffusion_active()
    myDiffNum.calculate_active_diff_list()
    
    with open("data.pkl", "wb") as f:
        pickle.dump(myDiffNum, f)
    
if __name__ == "__main__":
    main()