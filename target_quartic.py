from abstract_classes.target import TargetAbstract
import torch
import numpy as np
import scipy.interpolate


class TargetQuartic(TargetAbstract):
    def __init__(self, name="quartic", a=None, b=None, dim=None):

        super().__init__(name=name, dim=dim)

        self.a = a
        self.b = b

        if (self.a is not None) and (self.b is not None) and \
                (self.dim is not None):
            self.gen_target_sample()

    def gen_target_sample(self, a=None, b=None, dim=None, num_points=50000):
        a = self.a if a is None else a
        self.a = a

        b = self.b if b is None else b
        self.b = b

        dim = self.dim if dim is None else dim
        self.dim = dim

        xmin = -2*np.sqrt(abs(b/a))
        xmax = 2*np.sqrt(abs(b/a))

        x_arr = np.linspace(xmin, xmax, num_points)
        y_arr = -(a*x_arr**4 + b*x_arr**2)

        y_arr -= np.min(y_arr)
        y_arr = y_arr/(np.sum(y_arr))

        y_arr[0] = 0

        cdf_arr = 0*y_arr

        for i in range(1, num_points):
            cdf_arr[i] = cdf_arr[i-1] + y_arr[i]

        self.sample = torch.tensor(self.inverse_transform_sampling(
            x_arr, cdf_arr, self.dim))

    def inverse_transform_sampling(self, x, y, num_points):
        inv_cdf = scipy.interpolate.interp1d(y, x)
        r = np.random.rand(num_points)

        return inv_cdf(r)


if __name__ == "__main__":

    a = 1
    b = -10

    dim = 1000

    # Setting target parameters after creating the target object

    myTarget = TargetQuartic()

    myTarget.gen_target_sample(a=a, b=b, dim=dim)

    # Passing target parameters directly to target object constructor

    myTarget = TargetQuartic(a=a, b=b, dim=dim)
    
    # Plot a histogram of the target sample and save to file

    myTarget.plot_target_hist(fname="quartic_target_example.png",
                              title=f"a={a}, b={b}, dim={dim}")
