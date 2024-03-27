from diffusion_analytic import DiffusionAnalytic
from noise import NoiseActive, NoisePassive
from target_multi_gaussian import TargetMultiGaussian
from data_proc import DataProc

import numpy as np


class DiffusionAnalyticTest_Factory:
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

    reverse_diffusion_time = 0.5

    def create_test_objects(self):
        myPassiveNoise = NoisePassive(T=self.passive_noise_T,
                                      dim=self.sample_dim)

        myActiveNoise = NoiseActive(Tp=self.active_noise_Tp,
                                    Ta=self.active_noise_Ta,
                                    tau=self.tau,
                                    dim=self.sample_dim)

        myTarget = TargetMultiGaussian(mu_list=self.mu_list,
                                       sigma_list=self.sigma_list,
                                       pi_list=self.pi_list,
                                       dim=self.sample_dim,
                                       xmin=self.xmin,
                                       xmax=self.xmax)

        myDataProc = DataProc(xmin=self.xmin, xmax=self.xmax)

        myDiffNum = DiffusionAnalytic(ofile_base=self.ofile_base,
                                      passive_noise=myPassiveNoise,
                                      active_noise=myActiveNoise,
                                      target=myTarget,
                                      num_diffusion_steps=self.num_diffusion_steps,
                                      dt=self.dt,
                                      sample_dim=self.sample_dim,
                                      data_proc=myDataProc)

        return myDiffNum


def test_init():
    myFactory = DiffusionAnalyticTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    assert myDiffNum.sample_dim == myFactory.sample_dim
    assert myDiffNum.ofile_base == myFactory.ofile_base
    assert myDiffNum.num_diffusion_steps == myFactory.num_diffusion_steps
    assert myDiffNum.dt == myFactory.dt

    # Move to target test file
    assert myDiffNum.target.mu_list == myFactory.mu_list
    assert myDiffNum.target.sigma_list == myFactory.sigma_list
    assert myDiffNum.target.pi_list == myFactory.pi_list
    assert len(myDiffNum.target.sample) == myFactory.sample_dim


def test_forward_diffusion_passive():
    myFactory = DiffusionAnalyticTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    forward_samples_passive = myDiffNum.forward_diffusion_passive()

    assert isinstance(forward_samples_passive, list)
    assert len(forward_samples_passive) == myFactory.num_diffusion_steps + 1

    for sample in forward_samples_passive:
        assert isinstance(sample, np.ndarray)
        assert len(sample) == myFactory.sample_dim


def test_sample_from_diffusion_passive():
    myFactory = DiffusionAnalyticTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    x, reverse_diffusion_passive_samples = myDiffNum.sample_from_diffusion_passive(
        time=myFactory.reverse_diffusion_time)

    assert isinstance(x, np.ndarray)
    assert x.shape == (myFactory.sample_dim, )

    assert isinstance(reverse_diffusion_passive_samples, list)
    assert len(
        reverse_diffusion_passive_samples) == myFactory.num_diffusion_steps - 1

    for sample_t in reverse_diffusion_passive_samples:
        assert isinstance(sample_t, np.ndarray)
        assert sample_t.shape == (myFactory.sample_dim,)


def test_M_11_12_22():
    myFactory = DiffusionAnalyticTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    t_idx = 1

    M11, M12, M22 = myDiffNum.M_11_12_22(t_idx)

    assert isinstance(M11, float)
    assert isinstance(M12, float)
    assert isinstance(M22, float)


def test_score_function_active():
    myFactory = DiffusionAnalyticTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    x = np.sqrt(myDiffNum.active_noise.temperature.passive/myDiffNum.k +
                (myDiffNum.active_noise.temperature.active /
                 (myDiffNum.k**2 * myDiffNum.active_noise.correlation_time + myDiffNum.k)
                 )
                ) * np.random.randn(myDiffNum.sample_dim)

    eta = np.sqrt(myDiffNum.active_noise.temperature.active /
                  myDiffNum.active_noise.correlation_time) * \
        np.random.randn(myDiffNum.sample_dim)

    time_now = 0.5

    Fx, Feta = myDiffNum.score_function_active(x=x, eta=eta, t=time_now)

    assert isinstance(Fx, np.ndarray)
    assert Fx.shape == (myDiffNum.sample_dim,)
    assert isinstance(Feta, np.ndarray)
    assert Feta.shape == (myDiffNum.sample_dim,)


def test_sample_from_diffusion_active():
    myFactory = DiffusionAnalyticTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    time = 0.5

    x, eta, reverse_diffusion_active_samples_x, reverse_diffusion_active_samples_eta = \
        myDiffNum.sample_from_diffusion_active(time=time)

    assert isinstance(x, np.ndarray)
    assert x.shape == (myDiffNum.sample_dim,)

    assert isinstance(eta, np.ndarray)
    assert eta.shape == (myDiffNum.sample_dim,)

    assert isinstance(reverse_diffusion_active_samples_x, list)
    assert len(
        reverse_diffusion_active_samples_x) == myDiffNum.num_active_reverse_diffusion_steps
    for sample_t in reverse_diffusion_active_samples_x:
        assert isinstance(sample_t, np.ndarray)
        assert sample_t.shape == (myFactory.sample_dim,)

    assert isinstance(reverse_diffusion_active_samples_eta, list)
    assert len(
        reverse_diffusion_active_samples_eta) == myDiffNum.num_active_reverse_diffusion_steps

    for sample_t in reverse_diffusion_active_samples_eta:
        assert isinstance(sample_t, np.ndarray)
        assert sample_t.shape == (myFactory.sample_dim,)
