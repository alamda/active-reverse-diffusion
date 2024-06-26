import sys

sys.path.insert(0, '../src/')

from diffusion_numeric import DiffusionNumeric
from diffusion_analytic_1D import DiffusionAnalytic
from noise import NoiseActive, NoisePassive
from target_multi_gaussian_1D import TargetMultiGaussian
from target_quartic_1D import TargetQuartic
from data_proc import DataProc

import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import multiprocess
from multiprocess import Pool
import tqdm

from read_configs_1D import Configs

if __name__ == "__main__":
    myConfigs = Configs(filename='diffusion.conf')

    if not os.path.isfile(f"{myConfigs.ofile_base}.pkl"):

        myPassiveNoise = NoisePassive(T=myConfigs.passive_noise_T,
                                      dim=myConfigs.sample_dim)

        myActiveNoise = NoiseActive(Tp=myConfigs.active_noise_Tp,
                                    Ta=myConfigs.active_noise_Ta,
                                    tau=myConfigs.active_noise_tau,
                                    dim=myConfigs.sample_dim)

        if myConfigs.target_type in ('gaussian', 'Gaussian'):

            myTarget = TargetMultiGaussian(mu_list=myConfigs.mu_list,
                                           sigma_list=myConfigs.sigma_list,
                                           pi_list=myConfigs.pi_list,
                                           sample_size=myConfigs.sample_dim,
                                           xmin=myConfigs.xmin,
                                           xmax=myConfigs.xmax)

        elif myConfigs.target_type in ('double_well'):

            myTarget = TargetQuartic(a=myConfigs.a,
                                     b=myConfigs.b,
                                     xmin=myConfigs.xmin,
                                     xmax=myConfigs.xmax,
                                     sample_size=myConfigs.sample_dim)

        myDataProc = DataProc(xmin=myConfigs.xmin, xmax=myConfigs.xmax)

        if myConfigs.diffusion_calculation_type in ('numeric'):
            myDiffNum = DiffusionNumeric(ofile_base=myConfigs.ofile_base,
                                         passive_noise=myPassiveNoise,
                                         active_noise=myActiveNoise,
                                         target=myTarget,
                                         num_diffusion_steps=myConfigs.num_diffusion_steps,
                                         dt=myConfigs.dt,
                                         data_proc=myDataProc)

            if myConfigs.passive_training_iterations is not None:
                myDiffNum.train_diffusion_passive(
                    iterations=myConfigs.passive_training_iterations)
            else:
                myDiffNum.train_diffusion_passive(iterations=500)

            if myConfigs.active_training_iterations is not None:
                myDiffNum.train_diffusion_active(
                    iterations=myConfigs.active_training_iterations)
            else:
                myDiffNum.train_diffusion_active(iterations=1000)

        # elif myConfigs.diffusion_calculation_type in ('analytic'):
        #     myDiffNum = DiffusionAnalytic(ofile_base=myConfigs.ofile_base,
        #                                   passive_noise=myPassiveNoise,
        #                                   active_noise=myActiveNoise,
        #                                   target=myTarget,
        #                                   num_diffusion_steps=myConfigs.num_diffusion_steps,
        #                                   dt=myConfigs.dt,
        #                                   sample_so=myConfigs.sample_dim,
        #                                   data_proc=myDataProc)

        myDiffNum.sample_from_diffusion_passive()
        myDiffNum.calculate_passive_diff_list()

        myDiffNum.sample_from_diffusion_active()
        myDiffNum.calculate_active_diff_list()

        with open(f"{myDiffNum.ofile_base}.pkl", 'wb') as f:
            pickle.dump(myDiffNum, f)
    else:
        with open(f"{myConfigs.ofile_base}.pkl", 'rb') as f:
            myDiffNum = pickle.load(f)

            myDiffNum.data_proc.num_hist_bins = 20

            myDiffNum.calculate_passive_diff_list()
            myDiffNum.calculate_active_diff_list()

    np.savetxt('passive_diff_list.txt', np.array(
        myDiffNum.passive_diff_list), delimiter=' ')

    np.savetxt('active_diff_list.txt', np.array(
        myDiffNum.active_diff_list), delimiter=' ')
