from abstract_classes.diffusion_analytic import DiffusionAnalytic
import numpy as np


class DiffusionAnalyticPassive(DiffusionAnalytic):
    def __init__(self, name="diffAP", total_time=None, dt=None, num_steps=None,
                 noise_list=None, target=None, dim=None):

        super().__init__(name=name, total_time=total_time, dt=dt,
                         num_steps=num_steps, noise_list=noise_list,
                         target=target, dim=dim)

        self.data = self.initialize_data()

        self.score_fn_list = [self.score_fn]

    def initialize_data(self):
        noise = self.noise_list[0]
        return np.sqrt(noise.temperature)*np.random.randn(self.dim)

    def score_fn(self, time=None, noise_obj=None):
        a = np.exp(-time)
        Delta = noise_obj.temperature * (1-np.exp(-2*time))

        score_num = 0
        score_den = 0

        for (mu, sigma, weight) in zip(self.target.mu_list,
                                       self.target.sigma_list,
                                       self.target.weight_list):
            h = sigma*sigma

            Delta_eff = a*a*h + Delta

            z = np.exp((-(self.data - a*mu)**2)/(2*Delta_eff))

            score_num = score_num - weight * \
                np.sqrt(h)*np.power(Delta_eff, -1.5)*(self.data-a*mu)*z

            score_den = score_den + weight * \
                np.sqrt(h)*np.power(Delta_eff, -0.5)*z

        score = score_num / score_den

        return score


if __name__ == "__main__":
    from noise_passive import NoisePassive
    from target_multi_gaussian import TargetMultiGaussian

    import matplotlib.pyplot as plt

    dt = 0.02
    num_steps = 25
    dim = 10000

    myPassiveNoise = NoisePassive(T=1.0, dim=dim)
    myPassiveNoise.initialize_noise(dt=dt)
    noise_list = [myPassiveNoise]

    sigma_list = [1.0, 1.0, 1.0]
    mu_list = [-2.0, 0.0, 2.0]
    weight_list = [0.2, 0.5, 0.3]

    myTargetDsn = TargetMultiGaussian(sigma_list=sigma_list,
                                      mu_list=mu_list,
                                      weight_list=weight_list,
                                      dim=dim)
    myTargetDsn.sample()

    myDiff = DiffusionAnalyticPassive(dt=dt, num_steps=num_steps,
                                      noise_list=noise_list,
                                      target=myTargetDsn, dim=dim)

    data_trj = myDiff.reverse_diffuse()

    fig, ax = plt.subplots()

    ax.hist(myTargetDsn.samples.numpy(), bins=100, alpha=0.5)
    ax.hist(data_trj[-1][1], bins=100, alpha=0.5)

    plt.show()

    del myDiff
