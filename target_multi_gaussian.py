from abstract_classes.target import TargetAbstract
import torch
import numpy as np


class TargetMultiGaussian(TargetAbstract):
    def __init__(self, name="multi_gauss", sigma_list=None, mu_list=None, pi_list=None, dim=None):

        super().__init__(name=name, dim=dim)

        self.sigma_list = sigma_list
        self.mu_list = mu_list
        self.pi_list = pi_list

        if (self.sigma_list is not None) and \
            (self.mu_list is not None) and \
            (self.pi_list is not None) and \
                (self.dim is not None):

            self.gen_target_sample()

    def gen_target_sample(self, sigma_list=None, mu_list=None, pi_list=None, dim=None):
        sigma_list = self.sigma_list if sigma_list is None else sigma_list
        self.sigma_list = sigma_list

        mu_list = self.mu_list if mu_list is None else mu_list
        self.mu_list = mu_list

        pi_list = self.pi_list if pi_list is None else pi_list
        self.pi_list = pi_list

        dim = self.dim if dim is None else dim
        self.dim = dim

        dsn = torch.distributions.mixture_same_family.MixtureSameFamily(
            torch.distributions.Categorical(torch.tensor(pi_list)),
            torch.distributions.Normal(torch.tensor(
                mu_list), torch.tensor(sigma_list))
        )

        self.sample = dsn.sample(torch.Size([dim, 1]))


if __name__ == "__main__":

    sigma_list = [1.0, 1.0]
    mu_list = [-1.2, 1.2]
    pi_list = [1.0, 1.0]

    dim = 50000

    # Setting target parameters after creating the target object

    myTarget = TargetMultiGaussian()

    myTarget.gen_target_sample(sigma_list=sigma_list,
                               mu_list=mu_list,
                               pi_list=pi_list,
                               dim=dim)

    # Passing target parameters directly to target object constructor

    myTarget = TargetMultiGaussian(sigma_list=sigma_list,
                                   mu_list=mu_list,
                                   pi_list=pi_list,
                                   dim=dim)

    # Plot a histogram to target sample and save to file

    myTarget.plot_target_hist(fname="multi_gauss_target_example.png",
                              title=f"sigma={sigma_list}, mu={mu_list}, pi={pi_list}, dim={dim}",
                              range=(-5, 5))
