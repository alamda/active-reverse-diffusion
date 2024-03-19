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

import cProfile
import pickle
import os


sigmalist = [1.0, 1.0]  # Standard deviations of the distributions
mulist = [-1.2, 1.2]  # Means of the distributions
plist = [1.0, 1.0]  # Relative weight of the distributions

tsteps = 500  # Number of timesteps for running simulation
dt = 0.005  # Timestep size
T = 1.0  # Temperature for passive diffusion
Tp = 0  # Passive Temperature for Active diffusion
Ta = 1.0  # Active Temperature for Active diffusion
tau = 0.25  # Persistence Time
k = 1.0  # Not very relevant, just set it to 1
N = 50000  # Number of trajectories to be generated after training

data_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
    torch.distributions.Categorical(torch.tensor(plist)),
    torch.distributions.Normal(torch.tensor(mulist), torch.tensor(sigmalist))
)

dataset = data_distribution.sample(torch.Size([N, 1]))

with open('dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)

# Passive Case analytical
if os.path.isfile('x_passive.pkl'):
    with open('x_passive.pkl', 'rb') as f:
        x = pickle.load(f)
else:
    x = np.sqrt(T)*np.random.randn(N)

    with open('x_passive.pkl', 'wb') as f:
        pickle.dump(x, f)

if os.path.isfile('samples_passive_analytical.pkl'):
    with open('samples_passive_analytical.pkl', 'rb') as f:
        samples_passive_analytical = pickle.load(f)
else:
    samples_passive_analytical = rda.reverse_process_passive_new_multiple(
        x, T, tsteps, dt, sigmalist, mulist, plist)

    with open('samples_passive_analytical.pkl', 'wb') as f:
        pickle.dump(samples_passive_analytical, f)

if os.path.isfile('difflist_passive_analytical.pkl'):
    with open('difflist_passive_analytical.pkl', 'rb') as f:
        difflist_passive_analytical = pickle.load(f)
else:
    difflist_passive_analytical = ProcessData.diff_list(
        dataset, samples_passive_analytical, tsteps, xmin=-10, xmax=10, bandwidth=0.2, kernel='gaussian', npoints=1000)

    with open('difflist_passive_analytical.pkl', 'wb') as f:
        pickle.dump(difflist_passive_analytical, f)

# Passive Case numerical

if os.path.isfile('all_models_passive.pkl'):
    with open('all_models_passive.pkl', 'rb') as f:
        all_models_passive = pickle.load(f)
else:
    all_models_passive = rdn.passive_training(
        dataset, tsteps, T, dt, nrnodes=4, iterations=500)

    with open('all_models_passive.pkl', 'wb') as f:
        pickle.dump(all_models_passive, f)

if os.path.isfile('samples_passive_numerical.pkl'):
    with open('samples_passive_numerical.pkl', 'rb') as f:
        samples_passive_numerical = pickle.load(f)
else:
    _, samples_passive_numerical = rdn.sampling(
        N, all_models_passive, T, dt, tsteps)

    with open('samples_passive_numerical.pkl', 'wb') as f:
        pickle.dump(samples_passive_numerical, f)

if os.path.isfile('difflist_passive_numerical.pkl'):
    with open('difflist_passive_numerical.pkl', 'rb') as f:
        difflist_passive_numerical = pickle.load(f)
else:
    difflist_passive_numerical = ProcessData.diff_list(
        dataset, samples_passive_numerical, tsteps, xmin=-10, xmax=10, bandwidth=0.2, kernel='gaussian', npoints=1000)

    with open('difflist_passive_numerical.pkl', 'wb') as f:
        pickle.dump(difflist_passive_numerical, f)


# Active case analytical
if os.path.isfile('x_active.pkl'):
    with open('x_active.pkl', 'rb') as f:
        x = pickle.load(f)
else:
    x = np.sqrt(Tp/k + (Ta/(k*k*tau+k)))*np.random.randn(N)

    with open('x_active.pkl', 'wb') as f:
        pickle.dump(x, f)

if os.path.isfile('y_active.pkl'):
    with open('y_active.pkl', 'rb') as f:
        y = pickle.load(f)
else:
    y = np.sqrt(Ta/tau)*np.random.randn(N)

    with open('y_active.pkl', 'wb') as f:
        pickle.dump(y, f)

if os.path.isfile('samples_active_analytical.pkl'):
    with open('samples_active_analytical.pkl', 'rb') as f:
        samples_active_analytical = pickle.load(f)
else:
    samples_active_analytical, ymat = rda.reverse_process_active_new_multiple(
        x, y, Tp, Ta, tau, tsteps, dt, sigmalist, mulist, plist, k)

    with open('samples_active_analytical.pkl', 'wb') as f:
        pickle.dump(samples_active_analytical, f)

    with open('ymat.pkl', 'wb') as f:
        pickle.dump(ymat, f)

if os.path.isfile('difflist_active_analytical.pkl'):
    with open('difflist_active_analytical.pkl', 'rb') as f:
        difflist_active_analytical = pickle.load(f)
else:
    difflist_active_analytical = ProcessData.diff_list(
        dataset, samples_active_analytical, tsteps, xmin=-10, xmax=10, bandwidth=0.2, kernel='gaussian', npoints=1000)

    with open('difflist_active_analytical.pkl', 'wb') as f:
        pickle.dump(difflist_active_analytical, f)


# Active Case numerical
if os.path.isfile('all_models_x.pkl') and os.path.isfile('all_models_eta.pkl'):
    with open('all_models_x.pkl', 'rb') as f:
        all_models_x = pickle.load(f)

    with open('all_models_eta.pkl', 'rb') as f:
        all_models_eta = pickle.load(f)
else:
    all_models_x, all_models_eta = rdn.active_training(
        dataset, tsteps, Tp, Ta, tau, k, dt, nrnodes=4, iterations=1000)

    with open('all_models_x.pkl', 'wb') as f:
        pickle.dump(all_models_x, f)

    with open('all_models_eta.pkl', 'wb') as f:
        pickle.dump(all_models_eta, f)

if os.path.isfile('samples_active_numerical.pkl'):
    with open('samples_active_numerical.pkl', 'rb') as f:
        samples_active_numerical = pickle.load(f)
else:
    _, _, samples_active_numerical, _ = rdn.sampling_active(
        N, all_models_x, all_models_eta, Tp, Ta, tau, k, dt, tsteps)

    with open('samples_active_numerical.pkl', 'wb') as f:
        pickle.dump(samples_active_numerical, f)

if os.path.isfile('difflist_active_numerical.pkl'):
    with open('difflist_active_numerical.pkl', 'rb') as f:
        difflist_active_numerical = pickle.load(f)
else:
    difflist_active_numerical = ProcessData.diff_list(
        dataset, samples_active_numerical, tsteps, xmin=-10, xmax=10, bandwidth=0.2, kernel='gaussian', npoints=1000)

    with open('difflist_active_numerical.pkl', 'wb') as f:
        pickle.dump(difflist_active_numerical, f)

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlabel("Time")
ax.set_ylabel("Log(KL-Divergence)")
tlist = np.linspace(dt, tsteps*dt, tsteps-1)
ax.plot(tlist, np.log(difflist_passive_analytical), label="Passive-Analytical")
ax.plot(tlist, np.log(difflist_passive_numerical), label="Passive-Numerical")
ax.plot(tlist, np.log(difflist_active_analytical), label="Active-Analytical")
ax.plot(tlist, np.log(difflist_active_numerical), label="Active-Numerical")
ax.legend()

with open('fig.pkl', 'wb') as f:
    pickle.dump(fig, f)

with open('ax.pkl', 'wb') as f:
    pickle.dump(ax, f)

plt.savefig("diff.png")

plt.close(fig)
