from diffusion_numeric import DiffusionNumeric
from noise import NoiseActive, NoisePassive
from target_multi_gaussian import TargetMultiGaussian
from data_proc import DataProc

import torch


class DiffusionNumericTest_Factory:
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
                                       dim=self.sample_dim)

        myDataProc = DataProc(xmin=self.xmin, xmax=self.xmax)

        myDiffNum = DiffusionNumeric(ofile_base=self.ofile_base,
                                     passive_noise=myPassiveNoise,
                                     active_noise=myActiveNoise,
                                     target=myTarget,
                                     num_diffusion_steps=self.num_diffusion_steps,
                                     dt=self.dt,
                                     sample_dim=self.sample_dim,
                                     data_proc=myDataProc)

        return myDiffNum


def test_init():
    myFactory = DiffusionNumericTest_Factory()
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
    myFactory = DiffusionNumericTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    forward_samples_passive = myDiffNum.forward_diffusion_passive()

    assert isinstance(forward_samples_passive, list)
    assert len(forward_samples_passive) == myFactory.num_diffusion_steps + 1

    for sample in forward_samples_passive:
        assert isinstance(sample, torch.Tensor)
        assert len(sample) == myFactory.sample_dim


def test_compute_loss_passive():
    myFactory = DiffusionNumericTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    forward_samples_passive = myDiffNum.forward_diffusion_passive()

    nrnodes = 4

    score_model = torch.nn.Sequential(
        torch.nn.Linear(1, nrnodes), torch.nn.Tanh(),
        torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
        torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
        torch.nn.Linear(nrnodes, 1)
    ).double()

    t = 1

    loss, l, scr = myDiffNum.compute_loss_passive(
        forward_samples_passive, t, score_model)

    assert isinstance(loss, torch.Tensor)
    assert isinstance(l, torch.Tensor)
    assert isinstance(scr, torch.Tensor)


def test_train_diffusion_passive():
    myFactory = DiffusionNumericTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    passive_models = myDiffNum.train_diffusion_passive(iterations=10)

    assert isinstance(passive_models, list)
    assert len(passive_models) == myFactory.num_diffusion_steps - 1


def test_sample_from_diffusion_passive():
    myFactory = DiffusionNumericTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    forward_samples_passive = myDiffNum.forward_diffusion_passive()

    passive_models = myDiffNum.train_diffusion_passive(iterations=10)

    x, reverse_diffusion_passive_samples = myDiffNum.sample_from_diffusion_passive(
        passive_models)

    assert isinstance(x, torch.Tensor)
    assert x.shape == torch.Size([myFactory.sample_dim, 1])

    assert isinstance(reverse_diffusion_passive_samples, list)
    assert len(
        reverse_diffusion_passive_samples) == myFactory.num_diffusion_steps - 1

    for sample_t in reverse_diffusion_passive_samples:
        assert isinstance(sample_t, torch.Tensor)
        assert sample_t.shape == torch.Size([myFactory.sample_dim, 1])


def test_forward_diffusion_active():
    myFactory = DiffusionNumericTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    forward_samples_x, forward_samples_eta = myDiffNum.forward_diffusion_active()

    assert isinstance(forward_samples_x, list)
    assert len(forward_samples_x) == myFactory.num_diffusion_steps + 1

    assert isinstance(forward_samples_eta, list)
    assert len(forward_samples_eta) == myFactory.num_diffusion_steps + 1

    for sample in forward_samples_x:
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == torch.Size([myFactory.sample_dim, 1])

    for sample in forward_samples_eta:
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == torch.Size([myFactory.sample_dim, 1])


def test_M_11_12_22():
    myFactory = DiffusionNumericTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    t_idx = 1

    M11, M12, M22 = myDiffNum.M_11_12_22(t_idx)

    assert isinstance(M11, float)
    assert isinstance(M12, float)
    assert isinstance(M22, float)


def test_compute_loss_active():
    myFactory = DiffusionNumericTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    forward_samples_x, forward_samples_eta = myDiffNum.forward_diffusion_active()

    nrnodes = 4

    score_model_x = torch.nn.Sequential(
        torch.nn.Linear(2, nrnodes), torch.nn.Tanh(),
        torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
        torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
        torch.nn.Linear(nrnodes, 1)
    ).double()

    score_model_eta = torch.nn.Sequential(
        torch.nn.Linear(2, nrnodes), torch.nn.Tanh(),
        torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
        torch.nn.Linear(nrnodes, 1)
    ).double()

    t_idx = 1

    loss_x, loss_eta, loss_Fx, loss_Feta, loss_scr_x, loss_scr_eta = \
        myDiffNum.compute_loss_active(t_idx,
                                      forward_samples_x,
                                      forward_samples_eta,
                                      score_model_x,
                                      score_model_eta)

    assert isinstance(loss_x, torch.Tensor)
    assert isinstance(loss_eta, torch.Tensor)
    assert isinstance(loss_Fx, torch.Tensor)
    assert isinstance(loss_Feta, torch.Tensor)
    assert isinstance(loss_scr_x, torch.Tensor)
    assert isinstance(loss_scr_eta, torch.Tensor)


def test_train_diffusion_active():
    myFactory = DiffusionNumericTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    active_models_x, active_models_eta = myDiffNum.train_diffusion_active(
        iterations=10)

    assert isinstance(active_models_x, list)
    assert len(active_models_x) == myFactory.num_diffusion_steps - 1

    assert isinstance(active_models_eta, list)
    assert len(active_models_eta) == myFactory.num_diffusion_steps - 1


def test_sample_from_diffusion_active():
    myFactory = DiffusionNumericTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    active_models_x, active_models_eta = myDiffNum.train_diffusion_active(
        iterations=10)

    x, eta, reverse_diffusion_active_samples_x, reverse_diffusion_active_samples_eta = \
        myDiffNum.sample_from_diffusion_active(active_models_x,
                                               active_models_eta)

    assert isinstance(x, torch.Tensor)
    assert x.shape == torch.Size([myFactory.sample_dim, 1])

    assert isinstance(eta, torch.Tensor)
    assert eta.shape == torch.Size([myFactory.sample_dim, 1])

    assert isinstance(reverse_diffusion_active_samples_x, list)
    assert len(
        reverse_diffusion_active_samples_x) == myFactory.num_diffusion_steps - 1

    for sample_t in reverse_diffusion_active_samples_x:
        assert isinstance(sample_t, torch.Tensor)
        assert sample_t.shape == torch.Size([myFactory.sample_dim, 1])

    assert isinstance(reverse_diffusion_active_samples_eta, list)
    assert len(
        reverse_diffusion_active_samples_eta) == myFactory.num_diffusion_steps - 1

    for sample_t in reverse_diffusion_active_samples_eta:
        assert isinstance(sample_t, torch.Tensor)
        assert sample_t.shape == torch.Size([myFactory.sample_dim, 1])


def test_passive_and_active_diffusion():
    myFactory = DiffusionNumericTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    myDiffNum.train_diffusion_passive()
    myDiffNum.sample_from_diffusion_passive()
    myDiffNum.calculate_passive_diff_list()

    myDiffNum.train_diffusion_active()
    myDiffNum.sample_from_diffusion_active()
    myDiffNum.calculate_active_diff_list()

    # Passive
    assert isinstance(myDiffNum.passive_forward_samples, list)
    assert len(
        myDiffNum.passive_forward_samples) == myFactory.num_diffusion_steps + 1

    for sample in myDiffNum.passive_forward_samples:
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == torch.Size([myFactory.sample_dim, 1])

    assert isinstance(myDiffNum.passive_models, list)
    assert len(myDiffNum.passive_models) == myFactory.num_diffusion_steps - 1

    assert isinstance(myDiffNum.passive_reverse_samples, list)
    assert len(
        myDiffNum.passive_reverse_samples) == myFactory.num_diffusion_steps - 1

    assert isinstance(myDiffNum.passive_diff_list, list)
    assert len(myDiffNum.passive_diff_list) == \
        len(myDiffNum.passive_reverse_samples) - 1
    assert len(myDiffNum.passive_diff_list) == myFactory.num_diffusion_steps - 2

    # Active x
    assert isinstance(myDiffNum.active_forward_samples_x, list)
    assert len(
        myDiffNum.active_forward_samples_x) == myFactory.num_diffusion_steps + 1

    for sample in myDiffNum.active_forward_samples_x:
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == torch.Size([myFactory.sample_dim, 1])

    assert isinstance(myDiffNum.active_models_x, list)
    assert len(myDiffNum.active_models_x) == myFactory.num_diffusion_steps - 1

    assert isinstance(myDiffNum.active_reverse_samples_x, list)
    assert len(
        myDiffNum.active_reverse_samples_x) == myFactory.num_diffusion_steps - 1

    # Active eta
    assert isinstance(myDiffNum.active_forward_samples_eta, list)
    assert len(
        myDiffNum.active_forward_samples_eta) == myFactory.num_diffusion_steps + 1

    for sample in myDiffNum.active_forward_samples_eta:
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == torch.Size([myFactory.sample_dim, 1])

    assert isinstance(myDiffNum.active_models_eta, list)
    assert len(myDiffNum.active_models_eta) == myFactory.num_diffusion_steps - 1

    assert isinstance(myDiffNum.active_reverse_samples_eta, list)
    assert len(
        myDiffNum.active_reverse_samples_eta) == myFactory.num_diffusion_steps - 1

    # Active difflist (for x)
    assert isinstance(myDiffNum.active_diff_list, list)
    assert len(myDiffNum.active_diff_list) == \
        len(myDiffNum.active_reverse_samples_x) - 1
    assert len(myDiffNum.active_diff_list) == myFactory.num_diffusion_steps - 2
