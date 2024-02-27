from abstract_classes.target import TargetAbstract
import torch


class MultiGaussianTarget(TargetAbstract):
    def __init__(self, name="multi_gauss"):
        super().__init__(name=name)

    def define_target(self, sigma_list=[], mu_list=[], weight_list=[]):
        self.dsn = torch.distributions.mixture_same_family.MixtureSameFamily(
            torch.distributions.Categorical(torch.tensor(weight_list)),
            torch.distributions.Normal(torch.tensor(
                mu_list), torch.tensor(sigma_list))
        )

    def sample(self, num_samples):
        self.samples = self.dsn.sample(torch.Size([num_samples, 1]))


if __name__ == "__main__":

    myTarget = MultiGaussianTarget()

    sigma_list = [1.0, 1.0, 1.0]
    mu_list = [-2.0, 0.0, 2.0]
    weight_list = [0.2, 0.5, 0.3]

    num_samples = 10000

    myTarget.define_target(sigma_list=sigma_list,
                           mu_list=mu_list, weight_list=weight_list)

    myTarget.sample(num_samples=num_samples)

    del myTarget
