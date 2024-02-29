import numpy as np
import torch
import seaborn as sns
import itertools
from tqdm.auto import tqdm
import copy
import scipy.special as special
from sklearn.neighbors import KernelDensity
import ReverseDiffusionAnalytical as rda
import ReverseDiffusionNumerical as rdn
import ProcessData
import matplotlib.pyplot as plt

import pickle

import itertools
from itertools import permutations
import scipy.interpolate


def inverse_transform_sampling(x, y, n_samples):
    inv_cdf = scipy.interpolate.interp1d(y, x)
    r = np.random.rand(n_samples)
    return inv_cdf(r)


def diffuse_quartic(a=None, b=None, c=None, tsteps=None, dt=None,
                    Tp=None, Ta=None, tau=None, k=1, N=None):

    ofile_base = f"a{a}_b{b}_tsteps{tsteps}_dt{dt}_Tp{Tp}_Ta{Ta}_tau{tau}_N{N}"

    # sigmalist = [1.0, 1.0, 1.0]  # Standard deviations of the distributions
    # mulist = [-2.0, 0.0, 2.0]  # Means of the distributions
    # plist = [0.2, 0.5, 0.3]  # Relative weight of the distributions

    # data_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
    #     torch.distributions.Categorical(torch.tensor(plist)),
    #     torch.distributions.Normal(
    #         torch.tensor(mulist), torch.tensor(sigmalist))
    # )

    n = 50000
    x_arr = np.linspace(-20, 20, n)
    y_arr = a*x_arr**4 + b*x_arr**2 + c

    y_arr_u = y_arr

    y_arr[0] = 0
    y_arr = y_arr/(np.sum(y_arr))
    cdf_arr = 0*y_arr

    for i in range(1, n, 1):
        cdf_arr[i] = cdf_arr[i-1] + y_arr[i]

    dataset = torch.tensor(inverse_transform_sampling(
        x_arr, cdf_arr, N))

    fig, ax = plt.subplots()
    # ax.hist(dataset, bins=100, density=True, histtype='step')
    ax.hist(dataset, bins=100, density=True)
    ax.plot(x_arr, y_arr*1000, color='orange', alpha=0.50)

    plt.show()

    with open(f"{ofile_base}_target_sample.pkl", 'wb') as f:
        pickle.dump(dataset, f)

    # Passive Case numerical
    all_models_passive = rdn.passive_training(
        dataset, tsteps, T, dt, nrnodes=4, iterations=500)

    _, samples_passive_numerical = rdn.sampling(
        N, all_models_passive, T, dt, tsteps)

    with open(f"{ofile_base}_samples_PN.pkl", 'wb') as f:
        pickle.dump(samples_passive_numerical, f)

    difflist_passive_numerical = ProcessData.diff_list(
        dataset, samples_passive_numerical, tsteps, xmin=-10, xmax=10, bandwidth=0.2, kernel='gaussian', npoints=1000)

    with open(f"{ofile_base}_difflist_PN.pkl", 'wb') as f:
        pickle.dump(difflist_passive_numerical, f)

    # Active Case numerical
    all_models_x, all_models_eta = rdn.active_training(
        dataset, tsteps, Tp, Ta, tau, k, dt, nrnodes=4, iterations=500)

    _, _, samples_active_numerical, _ = rdn.sampling_active(
        N, all_models_x, all_models_eta, Tp, Ta, tau, k, dt, tsteps)

    with open(f"{ofile_base}_samples_AN.pkl", 'wb') as f:
        pickle.dump(samples_active_numerical, f)

    difflist_active_numerical = ProcessData.diff_list(
        dataset, samples_active_numerical, tsteps, xmin=-10, xmax=10, bandwidth=0.2, kernel='gaussian', npoints=1000)

    with open(f"{ofile_base}_difflist_AN.pkl", 'wb') as f:
        pickle.dump(difflist_active_numerical, f)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title("a={a}, b={b}, Ta={Ta}, Tp={Tp}, tau={tau}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Log(KL-Divergence)")
    tlist = np.linspace(dt, tsteps*dt, tsteps-1)
    ax.plot(tlist, np.log(difflist_passive_numerical),
            label="Passive-Numerical")
    ax.plot(tlist, np.log(difflist_active_numerical), label="Active-Numerical")
    ax.legend()

    plt.savefig(f"{ofile_base}_diff.png")

    fig.close()

    print(f"{ofile_Base} done")


tsteps = 25  # Number of timesteps for running simulation
dt = 0.02  # Timestep size
T = 1.0  # Temperature for passive diffusion
Tp = 0.5  # Passive Temperature for Active diffusion
Ta = 0.5  # Active Temperature for Active diffusion
tau = 0.2  # Persistence Time
k = 1.0  # Not very relevant, just set it to 1
N = 10000  # Number of trajectories to be generated after training


# a_list = np.linspace(1, 2, 3)
# b_list = np.linspace(1, 2, 3)

a_list = [1]
b_list = [-100]

for a in a_list:
    for b in b_list:
        c = b**2/(4*a)
        diffuse_quartic(a=a, b=b, c=c, tsteps=tsteps, dt=dt,
                        Tp=Tp, Ta=Ta, tau=tau, k=1, N=N)
