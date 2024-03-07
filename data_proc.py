import numpy as np
from sklearn.neighbors import KernelDensity
import torch
import scipy.special as special
import tqdm


class DataProc():
    def __init__(self, xmin=-10, xmax=10):
        self.xmin = xmin
        self.xmax = xmax

        self.t_list = None
        self.diff_list = None

    def approx_prob_dist(self, sample, sample_dim=None, bandwidth=0.2, kernel='gaussian'):
        model = KernelDensity(bandwidth=bandwidth, kernel=kernel)

        if type(sample) == torch.Tensor:
            model.fit(sample.numpy())
        else:
            model.fit(sample)

        if sample_dim is None:
            sample_dim = sample.shape[0]

        values = np.linspace(self.xmin, self.xmax, sample_dim)
        values = values.reshape((len(values), 1))

        log_probabilities = model.score_samples(values)

        probabilities = np.exp(log_probabilities)
        probabilities = probabilities/np.sum(probabilities)

        return values, probabilities

    def calc_KL_divergence(self, sample1, sample2):

        if len(sample1.shape) == 1:
            sample1 = sample1.reshape((sample1.shape[0], 1))

        if len(sample2.shape) == 1:
            sample2 = sample2.reshape((sample2.shape[0], 1))

        try:
            if sample1.shape == sample2.shape:
                sample_dim = sample1.shape[0]
            else:
                raise AssertionError
        except AssertionError:
            print(
                "sample1 and sample2 are not the same shape, cannot calculate KL divergence")

        _, h1 = self.approx_prob_dist(sample1, sample_dim=sample_dim)
        _, h2 = self.approx_prob_dist(sample2, sample_dim=sample_dim)

        diff = np.sum(special.rel_entr(h1, h2))

        return diff

    def calc_diff_vs_t(self, target_sample, diffusion_sample_list):
        t_list = []
        diff_list = []

        num_diffusion_steps = len(diffusion_sample_list)

        with tqdm.tqdm(total=len(proc_list)) as pbar:
            for t_idx in range(0, num_diffusion_steps-1):
                diff = self.calc_KL_divergence(target_sample,
                                            diffusion_sample_list[t_idx])
                t_list.append(t_idx)
                diff_list.append(diff)
                pbar.update()

        self.t_list = t_list
        self.diff_list = diff_list

        return diff_list

    def calc_diff_vs_t_multiproc(self, target_sample, diffusion_sample_list, pool=None):
        if pool is not None:
            print("Calculating KL divergences with multiprocessing enabled")
            proc_list = []

            diff_list = []

            num_diffusion_steps = len(diffusion_sample_list)

            for t_idx in range(num_diffusion_steps - 1):

                proc = pool.apply_async(self.calc_KL_divergence,
                                        (target_sample, diffusion_sample_list[t_idx]))

                proc_list.append(proc)

            with tqdm.tqdm(total=len(proc_list)) as pbar:
                for proc in proc_list:
                    diff_list.append(proc.get())
                    pbar.update()

            return diff_list
        else:
            print(
                "No pool object provided, not using multiprocessing to calculate KL divergences")

            return self.calc_diff_vs_t(target_sample, diffusion_sample_list)
