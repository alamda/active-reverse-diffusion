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



if __name__ == "__main__":
    ofile_base = "data"

    sample_dim = 50000

    passive_noise_T = 1.0

    active_noise_Tp = 0.0
    active_noise_Ta = 1.0
    active_noise_tau = 0.25

    mu_x_list = [-1.2, 1.2]
    mu_y_list = [0, 0]
    sigma_list = [1, 1]
    pi_list = [1, 1]

    xmin = -2
    xmax = 2
    ymin = -2
    ymax = 2

    num_diffusion_steps = 100
    dt = 0.005

    myPassiveNoise = NoisePassive(T=passive_noise_T,
                                  dim=sample_dim)

    myActiveNoise = NoiseActive(Tp=active_noise_Tp,
                                Ta=active_noise_Ta,
                                tau=active_noise_tau,
                                dim=sample_dim)

    myTarget = TargetMultiGaussian2D(mu_x_list=mu_x_list,
                                     mu_y_list=mu_y_list,
                                     sigma_list=sigma_list,
                                     pi_list=pi_list,
                                     dim=sample_dim,
                                     xmin=xmin, xmax=xmax,
                                     ymin=ymin, ymax=ymax)

    myTarget.gen_target_sample()

    myDataProc = DataProc(xmin=xmin, xmax=xmax)

    myDiffNum = DiffusionNumeric(ofile_base=ofile_base,
                                 passive_noise=myPassiveNoise,
                                 active_noise=myActiveNoise,
                                 target=myTarget,
                                 num_diffusion_steps=num_diffusion_steps,
                                 dt=dt,
                                 sample_dim=sample_dim,
                                 data_proc=myDataProc)
    
    breakpoint()
