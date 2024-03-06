import pickle
import os

import matplotlib.pyplot as plt

if __name__ == "__main__":
    fname = "abc.pkl"

    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            mydiff = pickle.load(f)

    fig, ax = plt.subplots()

    target_sample = mydiff.target.sample.flatten()
    passive_sample = mydiff.passive_reverse_samples[-1].flatten()
    active_sample = mydiff.active_reverse_samples_x[-1].flatten()

    hist, bins, _ = ax.hist((target_sample, passive_sample, active_sample),
                            bins=100,
                            density=True,
                            label=['target', 'passive', 'active'],
                            histtype='step',
                            fill=[True, False, False],
                            alpha=1)

    new_bins = (bins[1:] + bins[:-1])/2

    # ax.plot(new_bins, hist[0], label="target")
    # ax.plot(new_bins, hist[1], label="passive")
    # ax.plot(new_bins, hist[2], label="active")

    ax.legend()

    plt.show()
