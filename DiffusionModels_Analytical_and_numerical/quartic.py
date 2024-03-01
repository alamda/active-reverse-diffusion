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

import os
from multiprocess import Pool


def plot_hist(samples, x_arr, y_arr, png_fname, hist_fname, title):
    fig, ax = plt.subplots()
    hist, bins, _, = ax.hist(samples, bins=100, density=True)
    ax.plot(x_arr, y_arr*1000, color='orange', alpha=0.50)
    ax.set_title(title)

    plt.savefig(png_fname)

    with open(hist_fname, 'wb') as f:
        pickle.dump((hist, bins), f)

    plt.close(fig)


def plot_diff(difflist_passive_numerical, difflist_active_numerical,
              tlist, title, fname):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Log(KL-Divergence)")

    ax.plot(tlist, np.log(difflist_passive_numerical),
            label="Passive-Numerical")
    ax.plot(tlist, np.log(difflist_active_numerical), label="Active-Numerical")
    ax.legend()

    plt.savefig(fname)

    plt.close(fig)


def inverse_transform_sampling(x, y, n_samples):
    inv_cdf = scipy.interpolate.interp1d(y, x)
    r = np.random.rand(n_samples)
    return inv_cdf(r)


def diffuse_quartic(pool=None, a=None, b=None, c=None, tsteps=None, dt=None,
                    Tp=None, Ta=None, tau=None, k=1, N=None):

    ofile_base = f"a{a}_b{b}_tsteps{tsteps}_dt{dt}_Tp{Tp}_Ta{Ta}_tau{tau}_N{N}"

    ofile_target_sample = f"{ofile_base}_target_sample.pkl"
    pngfile_target_samples = f"{ofile_base}_target.png"
    histfile_target_samples = f"{ofile_base}_target_hist.pkl"

    ofile_samples_PN = f"{ofile_base}_samples_PN.pkl"
    pngfile_samples_PN = f"{ofile_base}_samples_PN.png"
    histfile_samples_PN = f"{ofile_base}_samples_PN_hist.pkl"
    ofile_difflist_PN = f"{ofile_base}_difflist_PN.pkl"

    ofile_samples_AN = f"{ofile_base}_samples_AN.pkl"
    pngfile_samples_AN = f"{ofile_base}_samples_AN.png"
    histfile_samples_AN = f"{ofile_base}_samples_AN_hist.pkl"
    ofile_difflist_AN = f"{ofile_base}_difflist_AN.pkl"

    ofile_difflist = f"{ofile_base}_difflist_compare.pkl"

    n = 50000
    x_arr = np.linspace(-20, 20, n)
    y_arr = a*x_arr**4 + b*x_arr**2 + c

    y_arr_u = y_arr

    y_arr[0] = 0
    y_arr = y_arr/(np.sum(y_arr))
    cdf_arr = 0*y_arr

    for i in range(1, n, 1):
        cdf_arr[i] = cdf_arr[i-1] + y_arr[i]

    if os.path.isfile(ofile_target_sample) and os.path.getsize(ofile_target_sample) > 0:
        print(f"loading {ofile_target_sample}")
        with open(ofile_target_sample, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = torch.tensor(inverse_transform_sampling(
            x_arr, cdf_arr, N))

        pool.apply_async(plot_hist, (dataset, x_arr, y_arr,
                                     pngfile_target_samples, histfile_target_samples,
                                     f"a={a}, b={b}, Ta={Ta}, Tp={Tp}, tau={tau} Target Sample",))

        dataset.reshape((N, 1))

        with open(ofile_target_sample, 'wb') as f:
            pickle.dump(dataset, f)

    # Passive Case numerical

    if os.path.isfile(ofile_samples_PN):
        print(f"loading {ofile_samples_PN}")
        with open(ofile_samples_PN, 'rb') as f:
            samples_passive_numerical = pickle.load(f)
    else:

        all_models_passive = rdn.passive_training(
            dataset, tsteps, T, dt, nrnodes=4, iterations=500)

        _, samples_passive_numerical = rdn.sampling(
            N, all_models_passive, T, dt, tsteps)

        with open(ofile_samples_PN, 'wb') as f:
            pickle.dump(samples_passive_numerical, f)

    if os.path.isfile(ofile_difflist_PN):
        print(f"loading {ofile_difflist_PN}")
        with open(ofile_difflist_PN, 'rb') as f:
            difflist_passive_numerical = pickle.load(f)
    else:
        difflist_passive_numerical = ProcessData.diff_list_multiproc(
            pool, dataset, samples_passive_numerical, tsteps, xmin=-10, xmax=10, bandwidth=0.2, kernel='gaussian', npoints=1000)

        with open(ofile_difflist_PN, 'wb') as f:
            pickle.dump(difflist_passive_numerical, f)

    # Active Case numerical
    if os.path.isfile(ofile_samples_AN):
        print(f"loading {ofile_samples_AN}")
        with open(ofile_samples_AN, 'rb') as f:
            samples_active_numerical = pickle.load(f)
    else:
        all_models_x, all_models_eta = rdn.active_training(
            dataset, tsteps, Tp, Ta, tau, k, dt, nrnodes=4, iterations=500)

        _, _, samples_active_numerical, _ = rdn.sampling_active(
            N, all_models_x, all_models_eta, Tp, Ta, tau, k, dt, tsteps)

        with open(ofile_samples_AN, 'wb') as f:
            pickle.dump(samples_active_numerical, f)

    if os.path.isfile(ofile_difflist_AN):
        print(f"loading {ofile_difflist_AN}")
        with open(ofile_difflist_AN, 'rb') as f:
            difflist_active_numerical = pickle.load(f)
    else:
        difflist_active_numerical = ProcessData.diff_list(
            dataset, samples_active_numerical, tsteps, xmin=-10, xmax=10, bandwidth=0.2, kernel='gaussian', npoints=1000)

        with open(ofile_difflist_AN, 'wb') as f:
            pickle.dump(difflist_active_numerical, f)

    tlist = np.linspace(dt, tsteps*dt, tsteps-1)

    title = f"a={a}, b={b}, Ta={Ta}, Tp={Tp}, tau={tau}"
    fname = f"{ofile_base}_diff.png"

    pool.apply_async(plot_diff, (difflist_passive_numerical, difflist_active_numerical,
                                 tlist, title, fname,))

    with open(ofile_difflist, 'wb') as f:
        pickle.dump((tlist, difflist_passive_numerical,
                    difflist_active_numerical), f)

    pool.close()
    pool.join()

    print(f"{ofile_base} done")


if __name__ == "__main__":

    tsteps = 500  # Number of timesteps for running simulation
    dt = 0.01  # Timestep size
    T = 1.0  # Temperature for passive diffusion
    Tp = 0.5  # Passive Temperature for Active diffusion
    Ta = 0.5  # Active Temperature for Active diffusion
    # tau = 0.2  # Persistence Time
    k = 1.0  # Not very relevant, just set it to 1
    N = 10000  # Number of trajectories to be generated after training

    # a_list = np.linspace(1, 2, 3)
    # b_list = np.linspace(1, 2, 3)

    a_list = [1]
    b_list = [-500, -300, -200, -100, -50, -10, -5, 0]
    tau_list = [0.01, 0.02, 0.05]

    with Pool(processes=16) as pool:
        for tau in tau_list:
            for a in a_list:
                for b in b_list:
                    ofile_base = f"a{a}_b{b}_tsteps{tsteps}_dt{dt}_Tp{Tp}_Ta{Ta}_tau{tau}_N{N}"

                    if not os.path.isfile(f"{ofile_base}_diff.png") and (a != 0):
                        c = b**2/(4*a)

                        print(ofile_base)

                        diffuse_quartic(pool=pool, a=a, b=b, c=c, tsteps=tsteps, dt=dt,
                                        Tp=Tp, Ta=Ta, tau=tau, k=1, N=N)

                        print()

                        exit()

    with Pool(processes=16) as pool:
        for tau in tau_list:
            for a in a_list:
                for b in b_list:
                    ofile_base = f"a{a}_b{b}_tsteps{tsteps}_dt{dt}_Tp{Tp}_Ta{Ta}_tau{tau}_N{N}"

                    ofile_samples_PN = f"{ofile_base}_samples_PN.pkl"
                    pngfile_samples_PN = f"{ofile_base}_samples_PN.png"
                    histfile_samples_PN = f"{ofile_base}_samples_PN_hist.pkl"

                    ofile_samples_AN = f"{ofile_base}_samples_AN.pkl"
                    pngfile_samples_AN = f"{ofile_base}_samples_AN.png"
                    histfile_samples_AN = f"{ofile_base}_samples_AN_hist.pkl"

                    n = 50000
                    x_arr = np.linspace(-20, 20, n)
                    y_arr = a*x_arr**4 + b*x_arr**2 + c

                    y_arr_u = y_arr

                    y_arr[0] = 0
                    y_arr = y_arr/(np.sum(y_arr))

                    if (os.path.isfile(ofile_samples_PN) and (not os.isfile(pngfile_samples_PN))):
                        with open(ofile_samples_PN, 'rb') as f:
                            samples_PN = pickle.load(f)

                        pool.apply_async(plot_hist, (samples_PN[-1].reshape(N), x_arr, y_arr,
                                                     pngfile_samples_PN, histfile_samples_PN,
                                                     f"a={a}, b={b}, Ta={Ta}, Tp={Tp}, tau={tau} Passive Sample",))

                    if (os.path.isfile(ofile_samples_AN) and (not os.isfile(pngfile_samples_AN))):
                        with open(ofile_samples_AN, 'rb') as f:
                            samples_PN = pickle.load(f)

                        pool.apply_async(plot_hist, (samples_AN[-1].reshape(N), x_arr, y_arr,
                                                     pngfile_samples_AN, histfile_samples_AN,
                                                     f"a={a}, b={b}, Ta={Ta}, Tp={Tp}, tau={tau} Passive Sample",))

        pool.close()
        pool.join()
