from data_proc import DataProc
from diffusion_numeric import DiffusionNumeric
from noise_passive import NoisePassive
from noise_active import NoiseActive
from target_multi_gaussian import TargetMultiGaussian

import numpy as np


class DataProcTest_Factory:
    sample_dim = 100
    ofile_base = "abc"
    num_diffusion_steps = 10
    dt = 0.01

    T_passive = 0.5
    T_active = 0.5
    tau = 0.1

    mu_list = [-2.0, 0.0, 2.0]
    sigma_list = [0.5, 0.5, 0.5]
    pi_list = [1.0, 1.0, 1.0]

    myPassiveNoise = NoisePassive(T=T_passive,
                                  dim=sample_dim)

    myActiveNoise = NoiseActive(T=T_active,
                                tau=tau,
                                dim=sample_dim)

    myTarget = TargetMultiGaussian(mu_list=mu_list,
                                   sigma_list=sigma_list,
                                   pi_list=pi_list,
                                   dim=sample_dim)

    myDiffNum = DiffusionNumeric(ofile_base=ofile_base,
                                 passive_noise=myPassiveNoise,
                                 active_noise=myActiveNoise,
                                 target=myTarget,
                                 num_diffusion_steps=num_diffusion_steps,
                                 dt=dt,
                                 sample_dim=sample_dim)

    xmin = -5
    xmax = 5

    def create_test_object(self):
        myDataProc = DataProc(xmin=self.xmin, xmax=self.xmax)

        return myDataProc


def test_init():
    myFactory = DataProcTest_Factory()
    myDataProc = myFactory.create_test_object()

    assert myDataProc.xmin == myFactory.xmin
    assert myDataProc.xmax == myFactory.xmax


def test_approx_prob_dist():
    myFactory = DataProcTest_Factory()
    myDataProc = myFactory.create_test_object()

    passive_models = myFactory.myDiffNum.train_diffusion_passive(iterations=10)

    x, reverse_diffusion_passive_samples = myFactory.myDiffNum.sample_from_diffusion_passive(
        passive_models)

    values, probabilities = \
        myDataProc.approx_prob_dist(reverse_diffusion_passive_samples[-1],
                                    sample_dim=myFactory.sample_dim)

    assert isinstance(values, np.ndarray)
    assert isinstance(probabilities, np.ndarray)


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
