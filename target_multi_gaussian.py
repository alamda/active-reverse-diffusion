from abstract_classes.target import TargetAbstract
import torch


class TargetMultiGaussian(TargetAbstract):
    def __init__(self, name="multi_gauss", sigma_list=None, mu_list=None, weight_list=None, dim=None):
        super().__init__(name=name, dim=dim)

        self.sigma_list = sigma_list
        self.mu_list = mu_list
        self.weight_list = weight_list

        self.define_target()

    def define_target(self, sigma_list=None, mu_list=None, weight_list=None, dim=None):
        if sigma_list is None:
            sigma_list = self.sigma_list
        else:
            self.sigma_list = sigma_list

        if mu_list is None:
            mu_list = self.mu_list
        else:
            self.mu_list = mu_list

        if weight_list is None:
            weight_list = self.weight_list
        else:
            self.weight_list = weight_list

        if dim is not None:
            self.dim = dim

        self.dsn = torch.distributions.mixture_same_family.MixtureSameFamily(
            torch.distributions.Categorical(torch.tensor(weight_list)),
            torch.distributions.Normal(torch.tensor(
                mu_list), torch.tensor(sigma_list))
        )

    def sample(self):
        try:
            if self.dsn is not None:
                self.samples = self.dsn.sample(torch.Size([self.dim, 1]))
            else:
                raise TypeError
        except TypeError:
            print("Target.dsn cannot be None (need to run Target.define_target)")


if __name__ == "__main__":

    myTarget = TargetMultiGaussian()

    sigma_list = [1.0, 1.0, 1.0]
    mu_list = [-2.0, 0.0, 2.0]
    weight_list = [0.2, 0.5, 0.3]

    num_samples = 10000

    myTarget.define_target(sigma_list=sigma_list,
                           mu_list=mu_list, weight_list=weight_list)

    myTarget.sample(num_samples=num_samples)

    del myTarget
