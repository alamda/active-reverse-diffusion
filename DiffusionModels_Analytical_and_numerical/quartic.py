import os
import pickle
import numpy as np
from multiprocess import Pool

from diffusion_test import generate_quartic_target, diffuse, plot_hist, plot_calculated_hist

if __name__ == "__main__":

    tsteps = 1000  # Number of timesteps for running simulation
    dt = 0.01  # Timestep size
    T = 1.0  # Temperature for passive diffusion
    Tp = 0.5  # Passive Temperature for Active diffusion
    Ta = 0.5  # Active Temperature for Active diffusion
    k = 1.0  # Not very relevant, just set it to 1
    N = 10000  # Number of trajectories to be generated after training

    a_list = [1]
    b_list = [-0.01, -5, -10, -20, -50]
    tau_list = [0.01, 0.02, 0.05]

    for tau in tau_list:
        for a in a_list:
            for b in b_list:
                ofile_base = f"a{a}_b{b}_tsteps{tsteps}_dt{dt}_Tp{Tp}_Ta{Ta}_tau{tau}_N{N}"

                if not os.path.isfile(f"{ofile_base}_difflist_compare.pkl") and (a != 0):
                    c = b**2/(4*a)

                    print(ofile_base)

                    plot_title = f"a={a}, b={b}, Ta={Ta}, Tp={Tp}, tau={tau} Target Sample"

                    title_str = f"a={a}, b={b}, Ta={Ta}, Tp={Tp}, tau={tau}"

                    dataset = generate_quartic_target(
                        ofile_base, N, a, b, plot_title=plot_title)

                    xmin = -2*np.sqrt(abs(b/a))
                    xmax = 2*np.sqrt(abs(b/a))

                    with Pool(processes=16) as pool:

                        diffuse(pool=pool, ofile_base=ofile_base, dataset=dataset,
                                tsteps=tsteps, dt=dt, Tp=Tp, Ta=Ta, tau=tau, N=N, xmin=xmin, xmax=xmax, title_str=title_str)

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

                    xmin = -2*np.sqrt(abs(b/a))
                    xmax = 2*np.sqrt(abs(b/a))

                    n = 50000
                    x_arr = np.linspace(xmin, xmax, n)
                    y_arr = -(a*x_arr**4 + b*x_arr**2)

                    y_arr -= np.min(y_arr)
                    y_arr = y_arr/(np.sum(y_arr))

                    if os.path.isfile(ofile_samples_PN):

                        title = f"a={a}, b={b}, Ta={Ta}, Tp={Tp}, tau={tau} Passive Sample"

                        if os.path.isfile(histfile_samples_PN):
                            with open(histfile_samples_PN, 'rb') as f:
                                hist, bins = pickle.load(f)

                            plot_calculated_hist(
                                hist, bins, pngfile_samples_PN, title)

                        else:
                            with open(ofile_samples_PN, 'rb') as f:
                                samples_PN = pickle.load(f)

                                pool.apply_async(plot_hist, (samples_PN[-1].reshape(N),
                                                             pngfile_samples_PN, histfile_samples_PN,
                                                             title,))

                    if os.path.isfile(ofile_samples_AN):

                        title = f"a={a}, b={b}, Ta={Ta}, Tp={Tp}, tau={tau} Active Sample"

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
