from diffusion_numeric import DiffusionNumeric
from noise_passive import NoisePassive
from noise_active import NoiseActive
from target_multi_gaussian import TargetMultiGaussian

import torch


class DiffusionNumericTest_Factory:
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

    def create_test_objects(self):
        myPassiveNoise = NoisePassive(T=self.T_passive,
                                      dim=self.sample_dim)

        myActiveNoise = NoiseActive(T=self.T_active,
                                    tau=self.tau,
                                    dim=self.sample_dim)

        myTarget = TargetMultiGaussian(mu_list=self.mu_list,
                                       sigma_list=self.sigma_list,
                                       pi_list=self.pi_list,
                                       dim=self.sample_dim)

        myDiffNum = DiffusionNumeric(ofile_base=self.ofile_base,
                                     passive_noise=myPassiveNoise,
                                     active_noise=myActiveNoise,
                                     target=myTarget,
                                     num_diffusion_steps=self.num_diffusion_steps,
                                     dt=self.dt,
                                     sample_dim=self.sample_dim)

        return myDiffNum


def test_init():
    myFactory = DiffusionNumericTest_Factory()
    myDiffNum = myFactory.create_test_objects()

    assert myDiffNum.sample_dim == myFactory.sample_dim
    assert myDiffNum.ofile_base == myFactory.ofile_base
    assert myDiffNum.num_diffusion_steps == myFactory.num_diffusion_steps
    assert myDiffNum.dt == myFactory.dt

    # Move to noise test file
    assert myDiffNum.passive_noise.temperature == myFactory.T_passive
    assert myDiffNum.passive_noise.dim == myFactory.sample_dim

    assert myDiffNum.active_noise.temperature == myFactory.T_active
    assert myDiffNum.active_noise.correlation_time == myFactory.tau
    assert myDiffNum.active_noise.dim == myFactory.sample_dim

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
