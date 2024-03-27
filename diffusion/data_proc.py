import numpy as np
from sklearn.neighbors import KernelDensity
import torch
import scipy.special as special
import tqdm


class DataProc():
    def __init__(self, xmin=-10, xmax=10, num_hist_bins=20):
        self.xmin = xmin
        self.xmax = xmax

        self.num_hist_bins = num_hist_bins

        self.t_list = None
        self.diff_list = None

        self.target_dsn = {'range': None,
                           'probabilities': None}

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

    def calc_KL_divergence(self, target_sample, test_sample):

        if len(target_sample.shape) == 1:
            target_sample = target_sample.reshape((target_sample.shape[0], 1))

        if len(test_sample.shape) == 1:
            test_sample = test_sample.reshape((test_sample.shape[0], 1))

        try:
            if target_sample.shape == test_sample.shape:
                sample_dim = target_sample.shape[0]
            else:
                raise AssertionError
        except AssertionError:
            print(
                "target_sample and test_sample are not the same shape, cannot calculate KL divergence")

        h_target, b_target = np.histogram(target_sample,
                                          bins=self.num_hist_bins,
                                          density='pdf',
                                          range=(self.xmin, self.xmax))

        b_target = (b_target[1:] + b_target[:-1])/2
        h_target = h_target / np.sum(h_target)

        # v2, h2 = self.approx_prob_dist(test_sample, sample_dim=sample_dim)

        h_test, b_test = np.histogram(test_sample,
                                      bins=self.num_hist_bins,
                                      density='pdf',
                                      range=(self.xmin, self.xmax))

        b_test = (b_test[1:] + b_test[:-1])/2
        h_test = h_test / np.sum(h_test)

        try:
            if not np.all(h_target >= 0):
                raise TypeError
        except TypeError:
            print("Target sample histogram must have positive values")

        try:
            if not np.all(h_test >= 0):
                raise TypeError
        except TypeError:
            print("Test sample histogram must have positive values")

        diff = np.sum(special.rel_entr(h_test, h_target))

        return diff

    def calc_diff_vs_t(self, target_sample, diffusion_sample_list):
        t_list = []
        diff_list = []

        num_diffusion_steps = len(diffusion_sample_list)

        with tqdm.tqdm(total=num_diffusion_steps) as pbar:
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
