import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import multiprocess
from multiprocess import Pool
import tqdm


def plot_hist(passive_sample, active_sample, target_sample,
              num_hist_bins, xmin, xmax,
              png_fname):
    fig, ax = plt.subplots()

    hist, bins, _ = ax.hist((passive_sample, active_sample, target_sample),
                            bins=num_hist_bins,
                            density=True,
                            label=['passive', 'active', 'target'],
                            histtype='step',
                            fill=False,
                            alpha=1,
                            range=(xmin, xmax))

    ax.set_ylim(top=0.5)

    ax.legend()

    plt.savefig(png_fname)
    plt.close(fig)

    return


def make_movie(sim_object=None, num_hist_bins=100):

    xmin = sim_object.data_proc.xmin
    xmax = sim_object.data_proc.xmax

    hist_max = 0

    target_sample = sim_object.target.sample.flatten()
    target_sample_array = target_sample.reshape(target_sample.shape[0])

    num_steps = len(mydiff.passive_reverse_samples)

    num_cpus = multiprocess.cpu_count()
    num_procs = num_cpus - 4

    with Pool(processes=num_procs) as pool:
        proc_list = []
        result_list = []

        for idx in range(num_steps):
            png_fname = f"hist{str(idx).zfill(3)}.png"

            passive_sample = sim_object.passive_reverse_samples[idx].flatten()
            passive_sample = passive_sample.reshape(passive_sample.shape[0])

            active_sample = sim_object.active_reverse_samples_x[idx].flatten()
            active_sample = active_sample.reshape(active_sample.shape[0])

            proc = pool.apply_async(plot_hist, (passive_sample, active_sample,
                                    target_sample_array, num_hist_bins, xmin, xmax, png_fname))

            proc_list.append(proc)

        print("Plotting histograms for each diffusion time step")
        with tqdm.tqdm(total=len(proc_list)) as pbar:
            for proc in proc_list:
                result_list.append(proc.get())
                pbar.update()


if __name__ == "__main__":
    fname = "data.pkl"

    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            mydiff = pickle.load(f)

    num_hist_bins = 100

    if False:
        make_movie(sim_object=mydiff, num_hist_bins=20)

    ###
    fig, ax = plt.subplots()

    ax.set_title("samples before diffusion")

    target_sample = mydiff.target.sample.flatten()
    passive_sample = mydiff.passive_reverse_samples[0].flatten()
    active_sample = mydiff.active_reverse_samples_x[0].flatten()

    hist, bins, _ = ax.hist((passive_sample, active_sample, target_sample),
                            bins=num_hist_bins,
                            density=True,
                            label=['passive', 'active', 'target'],
                            histtype='step',
                            fill=False,
                            alpha=1)

    new_bins = (bins[1:] + bins[:-1])/2

    ax.legend()

    plt.savefig("0hist.png")
    plt.close(fig)

    fig, ax = plt.subplots()

    ax.set_title("samples after diffusion")

    target_sample = mydiff.target.sample.flatten()
    passive_sample = mydiff.passive_reverse_samples[-1].flatten()
    active_sample = mydiff.active_reverse_samples_x[-1].flatten()

    hist, bins, _ = ax.hist((passive_sample, active_sample, target_sample),
                            bins=num_hist_bins,
                            density=True,
                            label=['passive', 'active', 'target'],
                            histtype='step',
                            fill=False,
                            alpha=1)

    new_bins = (bins[1:] + bins[:-1])/2

    ax.legend()

    plt.savefig("hist.png")

    plt.close(fig)

    fig, ax = plt.subplots()

    if mydiff.passive_diff_list is None:
        mydiff.calculate_passive_diff_list()

        with open(fname, 'wb') as f:
            pickle.dump(mydiff, f)

    if mydiff.active_diff_list is None:
        mydiff.calculate_active_diff_list()

        with open(fname, 'wb') as f:
            pickle.dump(mydiff, f)

    passive_diff_list = mydiff.passive_diff_list
    active_diff_list = mydiff.active_diff_list

    ax.set_xlabel("Time")
    ax.set_ylabel("Log(KL-Divergence)")

    ax.plot(np.log(passive_diff_list), label="passive")
    ax.plot(np.log(active_diff_list), label="active")

    ax.legend()

    plt.savefig("diff.png")

    plt.close(fig)
