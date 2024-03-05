import os
import pickle
import numpy as np
from multiprocess import Pool

from diffusion_test import generate_double_gaussian_target, diffuse, plot_hist, plot_calculated_hist


if __name__ == "__main__":
    tsteps = 1000  # Number of timesteps for running simulation
    dt = 0.01  # Timestep size
    T = 1.0  # Temperature for passive diffusion
    Tp = 0.5  # Passive Temperature for Active diffusion
    Ta = 0.5  # Active Temperature for Active diffusion
    k = 1.0  # Not very relevant, just set it to 1
    N = 10000  # Number of trajectories to be generated after training

    mu_iter_list = [0.1, 0.2, 0.5]
    sigma_iter_list = [0.1]
    pi_iter_list = [0.5]
    tau_list = [0.01, 0.02, 0.05]

    for mu in mu_iter_list:
        for sigma in sigma_iter_list:
            for pi in pi_iter_list:
                for tau in tau_list:
                    # Standard deviations of the distributions
                    sigma_list = [sigma, sigma]
                    mu_list = [-mu, mu]  # Means of the distributions
                    pi_list = [pi, pi]  # Relative weight of the distributions

                    ofile_base = f"m{mu}_s{sigma}_p{pi}_Ta{Ta}_Tp{Tp}_tau{tau}_N{N}"
                    title_str = f"mu={mu}, sigma={sigma}, tau={tau}, dt={dt}, N={N}"

                    if not os.path.isfile(f"{ofile_base}_difflist_compare.pkl"):
                        print(ofile_base)

                        xmin = -(mu+2*sigma)
                        xmax = mu+2*sigma

                        dataset = generate_double_gaussian_target(
                            ofile_base, N, mu_list, sigma_list, pi_list, plot_title="")

                        with Pool(processes=16) as pool:
                            diffuse(pool=pool, ofile_base=ofile_base, dataset=dataset, tsteps=tsteps,
                                    dt=dt, Tp=Tp, Ta=Ta, tau=tau, k=1, N=N, xmin=xmin, xmax=xmax, title_str=title_str)

                        print()

                        exit()

    with Pool(processes=16) as pool:
        for tau in tau_list:
            for mu in mu_iter_list:
                for sigma in sigma_iter_list:
                    for pi in pi_iter_list:
                        ofile_base = f"m{mu}_s{sigma}_p{pi}_Ta{Ta}_Tp{Tp}_tau{tau}_N{N}"
                        title = f"m{mu}_s{sigma}_p{pi}_Ta{Ta}_Tp{Tp}_tau{tau}_N{N}"

                        ofile_samples_PN = f"{ofile_base}_samples_PN.pkl"
                        pngfile_samples_PN = f"{ofile_base}_samples_PN.png"
                        histfile_samples_PN = f"{ofile_base}_samples_PN_hist.pkl"

                        ofile_samples_AN = f"{ofile_base}_samples_AN.pkl"
                        pngfile_samples_AN = f"{ofile_base}_samples_AN.png"
                        histfile_samples_AN = f"{ofile_base}_samples_AN_hist.pkl"

                        if os.path.isfile(ofile_samples_PN):

                            if os.path.isfile(histfile_samples_PN):
                                with open(histfile_samples_PN, 'rb') as f:
                                    hist, bins = pickle.load(f)

                                plot_calculated_hist(
                                    hist, bins, pngfile_samples_PN, title)

                            else:
                                with open(ofile_samples_PN, 'rb') as f:
                                    samples_PN = pickle.load(f)

                                    breakpoint()

                                    pool.apply_async(plot_hist, (samples_PN[-1].reshape(N),
                                                                 pngfile_samples_PN, histfile_samples_PN,
                                                                 title,))

                        if os.path.isfile(ofile_samples_AN):

                            if os.path.isfile(histfile_samples_AN):
                                with open(histfile_samples_AN, 'rb') as f:
                                    hist, bins = pickle.load(f)

                                plot_calculated_hist(
                                    hist, bins, pngfile_samples_AN, title)

                            else:
                                with open(ofile_samples_AN, 'rb') as f:
                                    samples_AN = pickle.load(f)

                                    pool.apply_async(plot_hist, (samples_AN[-1].reshape(N),
                                                                 pngfile_samples_AN, histfile_samples_AN,
                                                                 title,))
