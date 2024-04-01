from target import TargetAbstract

import torch
import numpy as np
import matplotlib.pyplot as plt


class TargetMultiGaussian2D(TargetAbstract):
    def __init__(self, name="multi_gauss_2d",
                 mu_x_list=None, mu_y_list=None,
                 sigma_list=None,
                 pi_list=None,
                 dim=None,
                 xmin=None, xmax=None,
                 ymin=None, ymax=None):

        super().__init__(name=name, dim=dim, xmin=xmin, xmax=xmax)

        self.ymin = ymin
        self.ymax = ymax

        self.mu_x_list = mu_x_list
        self.mu_y_list = mu_y_list
        self.sigma_list = sigma_list

        self.pi_list = pi_list

        param_list = [self.mu_x_list, self.mu_y_list,
                      self.sigma_list,
                      self.pi_list, self.dim]

        if all(val is not None for val in param_list):
            self.gen_target_sample()

    def gen_target_sample(self, mu_x_list=None, mu_y_list=None,
                          sigma_list=None,
                          pi_list=None,
                          dim=None,
                          num_points_x=50,
                          num_points_y=50,
                          xmin=None, xmax=None,
                          ymin=None, ymax=None):

        mu_x_list = self.mu_x_list if mu_x_list is None else mu_x_list
        self.mu_x_list = mu_x_list

        sigma_list = self.sigma_list if sigma_list is None else sigma_list
        self.sigma_list = sigma_list

        mu_y_list = self.mu_y_list if mu_y_list is None else mu_y_list
        self.mu_y_list = mu_y_list

        dim = self.dim if dim is None else dim
        self.dim = dim

        xmin = self.xmin if xmin is None else xmin
        self.xmin = xmin

        xmax = self.xmax if xmax is None else xmax
        self.xmax = xmax

        ymin = self.ymin if ymin is None else ymin
        self.ymin = ymin

        ymax = self.ymax if ymax is None else ymax
        self.ymax = ymax

        x_arr = np.linspace(self.xmin, self.xmax, num_points_x)
        y_arr = np.linspace(self.ymin, self.ymax, num_points_y)

        x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)

        z_arr = np.zeros((num_points_x, num_points_y))

        for mu_x, mu_y, sigma, pi \
                in zip(mu_x_list, mu_y_list, sigma_list, pi_list):

            for idx, x in np.ndenumerate(x_mesh):
                y = y_mesh[idx]

                z_arr[idx] += pi*np.exp(-1*((x-mu_x)**2 +
                                            (y-mu_y)**2) / 2*sigma**2)

        z_arr /= np.sum(np.sum(z_arr, axis=0), axis=0)

        self.x_arr = x_arr
        self.y_arr = y_arr
        self.prob_arr = z_arr

        fig, ax = plt.subplots()

        c = ax.pcolormesh(x_arr, y_arr, z_arr)
        ax.axis([x_arr.min(), x_arr.max(), y_arr.min(), y_arr.max()])
        fig.colorbar(c, ax=ax)

        plt.show()

        plt.close(fig)

        prob_arr_flat = z_arr.flatten()
        idx_list = [idx for idx, _ in np.ndenumerate(x_mesh)]
        idx_arr = np.array(idx_list)

        idx_samples = np.random.choice(len(idx_arr), self.dim, p=prob_arr_flat)

        x_samples = [x_arr[idx_arr[idx][0]] for idx in idx_samples]
        y_samples = [y_arr[idx_arr[idx][1]] for idx in idx_samples]

        # https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html

        hist, x_bins, y_bins = np.histogram2d(
            x_samples, y_samples, bins=50, density=True)

        # TODO: check if T is needed
        # hist = hist.T

        x_mg, y_mg = np.meshgrid(x_bins, y_bins)

        fig, ax = plt.subplots()

        plt.imshow(hist, extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])

        # ax.pcolormesh(x_mg, y_mg, hist)

        # ax.set_aspect('equal')

        plt.show()

        plt.close(fig)
