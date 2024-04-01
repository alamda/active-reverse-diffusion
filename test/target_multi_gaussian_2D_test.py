from .context import target_multi_gaussian_2D

from target_multi_gaussian_2D import TargetMultiGaussian2D


class TargetMultiGaussian2DTest_Factory:
    mu_x_list = [-1.2, 1.2]
    mu_y_list = [0, 0]
    sigma_list = [1, 1]
    pi_list = [1, 1]

    dim = 50000

    xmin = -2
    xmax = 2
    ymin = -2
    ymax = 2


def test_gen_target_sample():
    myFactory = TargetMultiGaussian2DTest_Factory()

    t = TargetMultiGaussian2D()

    t.gen_target_sample(mu_x_list=myFactory.mu_x_list,
                        mu_y_list=myFactory.mu_y_list,
                        sigma_list=myFactory.sigma_list,
                        pi_list=myFactory.pi_list,
                        dim=myFactory.dim,
                        xmin=myFactory.xmin,
                        xmax=myFactory.xmax,
                        ymin=myFactory.ymin,
                        ymax=myFactory.ymax)
