import sys

sys.path.insert(0, '/home/alexandral/git_repos/active-reverse-diffusion/src/')

import pickle
import numpy as np
from matplotlib import figure

with open("data.pkl", "rb") as f:
    data_obj = pickle.load(f)

idx_arr = data_obj.target.target_hist_idx_arr
prob_arr = data_obj.target.target_hist_idx_prob_arr

num_bins_x = np.unique(idx_arr[:,0]).shape[0]
num_bins_y = np.unique(idx_arr[:,1]).shape[0]

xmin = data_obj.target.xmin
xmax = data_obj.target.xmax
ymin = data_obj.target.ymin
ymax = data_obj.target.ymax

aspect = 1 #((xmax - xmin) / num_bins_x) / ((ymax - ymin) / num_bins_y)

fig = figure.Figure()
ax = fig.subplots(1)

im = ax.imshow(prob_arr.reshape(num_bins_x, num_bins_y),
               extent=[xmin, xmax, ymin, ymax],
               aspect=aspect)
fig.colorbar(im)

fig.savefig("prob.png")

print(prob_arr.sum())
