from .context import data_proc, diffusion_numeric, noise, target_multi_gaussian

from data_proc import DataProc
from diffusion_numeric import DiffusionNumeric
from noise import NoiseActive, NoisePassive
from target_multi_gaussian import TargetMultiGaussian

import numpy as np

from multiprocess import Pool


class DataProcTest_Factory:
    sample_dim = 100
    ofile_base = "abc"
    num_diffusion_steps = 10
    dt = 0.01

    passive_noise_T = 1.0

    active_noise_Tp = 0.5
    active_noise_Ta = 0.5
    tau = 0.1

    mu_list = [-2.0, 0.0, 2.0]
    sigma_list = [0.5, 0.5, 0.5]
    pi_list = [1.0, 1.0, 1.0]

    xmin = -5
    xmax = 5
    num_hist_bins = 20

    myPassiveNoise = NoisePassive(T=passive_noise_T,
                                  dim=sample_dim)

    myActiveNoise = NoiseActive(Tp=active_noise_Tp,
                                Ta=active_noise_Ta,
                                tau=tau,
                                dim=sample_dim)

    myTarget = TargetMultiGaussian(mu_list=mu_list,
                                   sigma_list=sigma_list,
                                   pi_list=pi_list,
                                   dim=sample_dim,
                                   xmin=xmin,
                                   xmax=xmax)

    myDiffNum = DiffusionNumeric(ofile_base=ofile_base,
                                 passive_noise=myPassiveNoise,
                                 active_noise=myActiveNoise,
                                 target=myTarget,
                                 num_diffusion_steps=num_diffusion_steps,
                                 dt=dt,
                                 sample_dim=sample_dim)

    def create_test_object(self):
        myDataProc = DataProc(xmin=self.xmin,
                              xmax=self.xmax,
                              num_hist_bins=self.num_hist_bins)

        return myDataProc


def test_init():
    myFactory = DataProcTest_Factory()
    myDataProc = myFactory.create_test_object()

    assert myDataProc.xmin == myFactory.xmin
    assert myDataProc.xmax == myFactory.xmax
    assert myDataProc.num_hist_bins == myFactory.num_hist_bins


def test_calc_KL_divergence():
    myFactory = DataProcTest_Factory()
    myDataProc = myFactory.create_test_object()

    passive_models = myFactory.myDiffNum.train_diffusion_passive(iterations=10)

    x, reverse_diffusion_passive_samples = myFactory.myDiffNum.sample_from_diffusion_passive(
        passive_models)

    diff = myDataProc.calc_KL_divergence(
        reverse_diffusion_passive_samples[-1], myFactory.myDiffNum.target.sample)

    assert isinstance(diff, float)


def test_calc_diff_vs_t():
    myFactory = DataProcTest_Factory()
    myDataProc = myFactory.create_test_object()

    passive_models = myFactory.myDiffNum.train_diffusion_passive(iterations=10)

    x, reverse_diffusion_passive_samples = myFactory.myDiffNum.sample_from_diffusion_passive(
        passive_models)

    difflist = myDataProc.calc_diff_vs_t(
        myFactory.myDiffNum.target.sample, reverse_diffusion_passive_samples)

    assert len(difflist) == len(reverse_diffusion_passive_samples) - 1
    assert len(difflist) == myFactory.num_diffusion_steps - 2


def test_calc_diff_vs_t_multiproc():
    myFactory = DataProcTest_Factory()
    myDataProc = myFactory.create_test_object()

    passive_models = myFactory.myDiffNum.train_diffusion_passive(iterations=10)

    x, reverse_diffusion_passive_samples = myFactory.myDiffNum.sample_from_diffusion_passive(
        passive_models)

    with Pool(processes=4) as pool:
        difflist = myDataProc.calc_diff_vs_t_multiproc(myFactory.myDiffNum.target.sample,
                                                       reverse_diffusion_passive_samples,
                                                       pool=pool)

    assert len(difflist) == len(reverse_diffusion_passive_samples) - 1
    assert len(difflist) == myFactory.num_diffusion_steps - 2
