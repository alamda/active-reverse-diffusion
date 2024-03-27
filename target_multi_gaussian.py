from target import TargetAbstract
import torch
import numpy as np


class TargetMultiGaussian(TargetAbstract):
    def __init__(self, name="multi_gauss", sigma_list=None, mu_list=None, pi_list=None, dim=None,
                 xmin=None, xmax=None):

        super().__init__(name=name, dim=dim, xmin=xmin, xmax=xmax)

        self.sigma_list = sigma_list
        self.mu_list = mu_list
        self.pi_list = pi_list

        if (self.sigma_list is not None) and \
            (self.mu_list is not None) and \
            (self.pi_list is not None) and \
                (self.dim is not None):

            self.gen_target_sample()

    def gen_target_sample(self, sigma_list=None, mu_list=None, pi_list=None, dim=None,
                          xmin=None, xmax=None, num_points=50000):
        sigma_list = self.sigma_list if sigma_list is None else sigma_list
        self.sigma_list = sigma_list

        mu_list = self.mu_list if mu_list is None else mu_list
        self.mu_list = mu_list

        pi_list = self.pi_list if pi_list is None else pi_list
        self.pi_list = pi_list

        dim = self.dim if dim is None else dim
        self.dim = dim

        self.xmin = xmin if xmin is not None else self.xmin

        self.xmax = xmax if xmax is not None else self.xmax

        x_arr = np.linspace(self.xmin, self.xmax, num_points)

        y_arr = np.zeros(num_points)

        for mu, sigma, pi in zip(mu_list, sigma_list, pi_list):
            y_arr += pi*np.exp(-1*(x_arr-mu)**2/(2*sigma**2))

        y_arr /= np.sum(y_arr)

        self.x_arr = x_arr
        self.prob_arr = y_arr

        self.sample = torch.from_numpy(
            np.random.choice(x_arr, size=self.dim, p=y_arr))


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
                               dim=dim,
                               xmin=-5,
                               xmax=5)

    # Passing target parameters directly to target object constructor

    myTarget = TargetMultiGaussian(sigma_list=sigma_list,
                                   mu_list=mu_list,
                                   pi_list=pi_list,
                                   dim=dim,
                                   xmin=-5,
                                   xmax=5)

    # Plot a histogram to target sample and save to file

    myTarget.plot_target_hist(fname="multi_gauss_target_example.png",
                              title=f"sigma={sigma_list}, mu={mu_list}, pi={pi_list}, dim={dim}",
                              hist_range=(-5, 5))
