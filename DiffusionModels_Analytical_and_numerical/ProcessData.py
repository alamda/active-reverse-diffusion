import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import itertools
from tqdm.auto import tqdm
import copy
import scipy.special as special
from sklearn.neighbors import KernelDensity


def approx_prob_dist(dataset, xmin, xmax, bandwidth, kernel, npoints=1000):
    model = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    if type(dataset) == torch.Tensor:
        model.fit(dataset.numpy())
    else:
        model.fit(dataset)
    values = np.linspace(xmin, xmax, npoints)
    values = values.reshape((len(values), 1))
    log_probabilities = model.score_samples(values)
    probabilities = np.exp(log_probabilities)
    probabilities = probabilities/np.sum(probabilities)
    return values, probabilities


# datasets are torch tensors
def KL_div_between_datasets(dataset1, dataset2, xmin, xmax, bandwidth, kernel, npoints=1000):

    if len(dataset1.shape) == 1:
        dataset1 = dataset1.reshape((dataset1.shape[0], 1))
    if len(dataset2.shape) == 1:
        dataset2 = dataset2.reshape((dataset2.shape[0], 1))

    _, h1 = approx_prob_dist(dataset1, xmin, xmax, bandwidth, kernel, npoints)
    _, h2 = approx_prob_dist(dataset2, xmin, xmax, bandwidth, kernel, npoints)
    diff = np.sum(special.rel_entr(h1, h2))
    return diff


def diff_list(orig_dataset, time_dataset, tsteps, xmin=-10, xmax=10, bandwidth=0.2, kernel='gaussian', npoints=1000):
    difflist = []
    for t in range(0, tsteps-1):
        dataset1 = orig_dataset
        if type(time_dataset) == list:
            dataset2 = time_dataset[t]  # For Numerical procedure
        else:
            z = time_dataset[:, tsteps-t-1]
            dataset2 = np.reshape(z, (len(z), 1))  # For analytical procedure
        diff = KL_div_between_datasets(
            dataset1, dataset2, xmin, xmax, bandwidth, kernel)
        difflist.append(diff)
    return difflist


def diff_list_multiproc(pool, orig_dataset, time_dataset, tsteps, xmin=-10, xmax=10, bandwidth=0.2, kernel='gaussian', npoints=1000):
    difflist = []
    proclist = []
    for t in range(0, tsteps-1):
        dataset1 = orig_dataset
        if type(time_dataset) == list:
            dataset2 = time_dataset[t]  # For Numerical procedure
        else:
            z = time_dataset[:, tsteps-t-1]
            dataset2 = np.reshape(z, (len(z), 1))  # For analytical procedure

        proc = pool.apply_async(KL_div_between_datasets,
                                (dataset1, dataset2, xmin, xmax, bandwidth, kernel,))
        proclist.append(proc)

    for proc in proclist:
        difflist.append(proc.get())

    return difflist
