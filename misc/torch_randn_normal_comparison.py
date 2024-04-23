import numpy as np
import torch
from matplotlib import figure

sample_size = 100000
sample_dim = 2
num_bins = 20

xmin = -2
xmax = 2
ymin = -2
ymax = 2

normal_tensor = torch.normal(torch.zeros(sample_size, sample_dim),
                             torch.ones(sample_size, sample_dim))

randn_tensor = torch.randn(sample_size, sample_dim)

normal_hist, _, _ = np.histogram2d(normal_tensor[:,0], normal_tensor[:,1],
                                   bins=num_bins,
                                   density=True, 
                                   range=[[xmin, xmax], [ymin, ymax]])

normal_hist_slice = normal_hist[num_bins//2,:]

randn_hist, _, _ = np.histogram2d(randn_tensor[:,0], randn_tensor[:,1],
                                  bins=num_bins,
                                  density=True,
                                  range=[[xmin, xmax], [ymin, ymax]])

randn_hist_slice = randn_hist[num_bins//2,:]

slice_x = np.linspace(xmin, xmax, num_bins)

fig = figure.Figure()
axs = fig.subplots(1,3)

img_normal = axs[0].imshow(normal_hist,
                           extent=[xmin, xmax, ymin, ymax])
axs[0].set_title("torch.normal")

img_randn = axs[1].imshow(randn_hist,
                          extent=[xmin, xmax, ymin, ymax])
axs[1].set_title("torch.randn")

axs[2].scatter(slice_x, normal_hist_slice, label="torch.normal")
axs[2].scatter(slice_x, randn_hist_slice, label="torch.randn")

axs[2].legend()



fig.savefig("hist.png")