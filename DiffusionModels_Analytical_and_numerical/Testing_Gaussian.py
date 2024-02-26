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

sigmalist = [1.0, 1.0, 1.0] # Standard deviations of the distributions
mulist = [-2.0, 0.0, 2.0] # Means of the distributions
plist = [0.2, 0.5, 0.3] # Relative weight of the distributions

tsteps = 25 # Number of timesteps for running simulation
dt = 0.02 # Timestep size
T = 1.0 # Temperature for passive diffusion
Tp = 0.5 # Passive Temperature for Active diffusion
Ta = 0.5 # Active Temperature for Active diffusion
tau = 0.2 # Persistence Time
k = 1.0 # Not very relevant, just set it to 1
N = 10000 # Number of trajectories to be generated after training

data_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
    torch.distributions.Categorical(torch.tensor(plist)),
    torch.distributions.Normal(torch.tensor(mulist), torch.tensor(sigmalist))
)

dataset = data_distribution.sample(torch.Size([N, 1]))

with cProfile.Profile() as pr:
    # Passive Case analytical
    x = np.sqrt(T)*np.random.randn(N)
    samples_passive_analytical = rda.reverse_process_passive_new_multiple(x, T, tsteps, dt, sigmalist, mulist, plist)
    difflist_passive_analytical = ProcessData.diff_list(dataset, samples_passive_analytical, tsteps, xmin=-10, xmax=10, bandwidth=0.2, kernel='gaussian', npoints=1000)

    pr.dump_stats("passive_analytical.prof")
    
with cProfile.Profile() as pr: 
    #Passive Case numerical
    all_models_passive = rdn.passive_training(dataset, tsteps, T, dt, nrnodes=4, iterations=500)
    _, samples_passive_numerical = rdn.sampling(N, all_models_passive, T, dt, tsteps)
    difflist_passive_numerical = ProcessData.diff_list(dataset, samples_passive_numerical, tsteps, xmin=-10, xmax=10, bandwidth=0.2, kernel='gaussian', npoints=1000)
    
    pr.dump_stats("passive_numerical.prof")


with cProfile.Profile() as pr: 
    # Active case analytical
    x = np.sqrt(Tp/k + (Ta/(k*k*tau+k)))*np.random.randn(N)
    y = np.sqrt(Ta/tau)*np.random.randn(N)
    samples_active_analytical, ymat = rda.reverse_process_active_new_multiple(x, y, Tp, Ta, tau, tsteps, dt, sigmalist, mulist, plist, k)
    difflist_active_analytical = ProcessData.diff_list(dataset, samples_active_analytical, tsteps, xmin=-10, xmax=10, bandwidth=0.2, kernel='gaussian', npoints=1000)

    pr.dump_stats("active_analytical.prof")
    
with cProfile.Profile() as pr: 
    # Active Case numerical
    all_models_x, all_models_eta = rdn.active_training(dataset, tsteps, Tp, Ta, tau, k, dt, nrnodes=4, iterations=500)
    _, _, samples_active_numerical, _ = rdn.sampling_active(N, all_models_x, all_models_eta, Tp, Ta, tau, k, dt, tsteps)
    difflist_active_numerical = ProcessData.diff_list(dataset, samples_active_numerical, tsteps, xmin=-10, xmax=10, bandwidth=0.2, kernel='gaussian', npoints=1000)

    pr.dump_stats("active_numerical.prof")

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlabel("Time")
ax.set_ylabel("Log(KL-Divergence)")
tlist = np.linspace(dt, tsteps*dt, tsteps-1)
ax.plot(tlist, np.log(difflist_passive_analytical), label="Passive-Analytical")
ax.plot(tlist, np.log(difflist_passive_numerical), label="Passive-Numerical")
ax.plot(tlist, np.log(difflist_active_analytical), label="Active-Analytical")
ax.plot(tlist, np.log(difflist_active_numerical), label="Active-Numerical")
ax.legend()

plt.savefig("diff.png")
