from abstract_classes.target import TargetAbstract
import torch
import numpy as np
import scipy.interpolate


class TargetQuartic(TargetAbstract):
    def __init__(self, name="quartic", a=None, b=None, dim=None, xmin=None, xmax=None):

        super().__init__(name=name, dim=dim, xmin=xmin, xmax=xmax)

        self.a = a
        self.b = b

        if (self.a is not None) and (self.b is not None) and \
                (self.dim is not None):
            self.gen_target_sample()

    def gen_target_sample(self, a=None, b=None, dim=None, num_points=50000, xmin=None, xmax=None):
        a = self.a if a is None else a
        self.a = a

        b = self.b if b is None else b
        self.b = b

        dim = self.dim if dim is None else dim
        self.dim = dim

        xmin = self.xmin if self.xmin is not None else -2*np.sqrt(abs(b/a))
        self.xmin = xmin

        xmax = self.xmax if self.xmax is not None else 2*np.sqrt(abs(b/a))
        self.xmax = xmax

        x_arr = np.linspace(xmin, xmax, num_points)
        y_arr = np.exp(-(a*x_arr**4 + b*x_arr**2))

        y_arr = y_arr/(np.sum(y_arr))

        self.x_arr = x_arr
        self.prob_arr = y_arr

        self.sample = torch.from_numpy(np.random.choice(x_arr, size=self.dim, p=y_arr))


if __name__ == "__main__":

    a = 0.035
    b = -0.1

    dim = 50000

    xmin = -5
    xmax = 5

    # Setting target parameters after creating the target object

    myTarget = TargetQuartic()

    myTarget.gen_target_sample(a=a, b=b, dim=dim)

    # Passing target parameters directly to target object constructor

    myTarget = TargetQuartic(a=a, b=b, dim=dim)

    # Plot a histogram of the target sample and save to file

    myTarget.plot_target_hist(fname="quartic_target_example.png",
                              title=f"a={a}, b={b}, dim={dim}",
                              hist_range=(-5, 5))
