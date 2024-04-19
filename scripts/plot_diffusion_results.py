import sys

sys.path.insert(0, '/home/alexandral/git_repos/active-reverse-diffusion/src/')

from matplotlib import figure
import numpy as np
import mmap
import pickle

addl_time_to_plot = "0.4" # set to none if not needed

def mmap_np(fname, shape=None):
    with open(fname, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        mm_arr = np.frombuffer(mm, dtype=np.double)
        
        if shape is None:
            shape = (-1, 2)
            
        mm_arr = np.reshape(mm_arr, shape)

    return mm, mm_arr

target_mm, target = mmap_np("target.npy")

sample_size = target.shape[0]
shape = (-1, sample_size, 2)

pf_mm, passive_forward = mmap_np("forward_passive.npy", shape=shape)
af_mm, active_forward = mmap_np("forward_x_active.npy", shape=shape)

pr_mm, passive_reverse = mmap_np("reverse_passive.npy", shape=shape)
ar_mm, active_reverse = mmap_np("reverse_x_active.npy", shape=shape)

with open("data.pkl", "rb") as f:
    data_obj = pickle.load(f)

xmin = data_obj.target.xmin
xmax = data_obj.target.xmax
ymin = data_obj.target.ymin
ymax = data_obj.target.ymax

extent = [xmin, xmax, ymin, ymax]

dt = data_obj.dt

time_f = dt * passive_forward.shape[0]
time_r = dt * passive_reverse.shape[0]

idx_arr = data_obj.target.target_hist_idx_arr
prob_arr = data_obj.target.target_hist_idx_prob_arr

num_bins_x = np.unique(idx_arr[:,0]).shape[0]
num_bins_y = np.unique(idx_arr[:,1]).shape[0]

t_hist, _, _ = np.histogram2d(target[:,0], target[:,1])

pf_f, _, _ = np.histogram2d(passive_forward[0,:,0], passive_forward[0,:,1])
pf_l, _, _ = np.histogram2d(passive_forward[-1,:,0], passive_forward[-1,:,1])
af_f, _, _ = np.histogram2d(active_forward[0,:,0], active_forward[0,:,1])
af_l, _, _ = np.histogram2d(active_forward[-1,:,0], active_forward[-1,:,1])

pr_f, _, _ = np.histogram2d(passive_reverse[0,:,0], passive_reverse[0,:,1])
pr_l, _, _ = np.histogram2d(passive_reverse[-1,:,0], passive_reverse[-1,:,1])
ar_f, _, _ = np.histogram2d(active_reverse[0,:,0], active_reverse[0,:,1])
ar_l, _, _ = np.histogram2d(active_reverse[-1,:,0], active_reverse[-1,:,1])

if isinstance(addl_time_to_plot, str) and len(addl_time_to_plot) > 0:
    addl_pass_fname = f"pass_rev_{addl_time_to_plot}.npy"
    addl_act_fname = f"act_rev_{addl_time_to_plot}.npy"

    time_c = float(addl_time_to_plot)

    pr_c_mm, pr = mmap_np(addl_pass_fname, shape=shape)
    ar_c_mm, ar = mmap_np(addl_act_fname, shape=shape)

    pr_c, _, _ = np.histogram2d(pr[-1,:,0], pr[-1,:,1])
    ar_c, _, _ = np.histogram2d(ar[-1,:,0], ar[-1,:,1])

fig = figure.Figure()


if isinstance(addl_time_to_plot, str) and len(addl_time_to_plot) > 0:
    axs = fig.subplots(4,5)
    fig.set_size_inches(12,6)
else:
    axs = fig.subplots(3,5)
    fig.set_size_inches(10,5)

axs[0,0].axis("off")
axs[0,1].axis("off")

im02 = axs[0,2].imshow(prob_arr.reshape(num_bins_x, num_bins_y),
                       extent=extent)
axs[0,2].set_title("t_prob")

axs[0,3].axis("off")
axs[0,4].axis("off")

im10 = axs[1,0].imshow(pf_f, extent=extent)
axs[1,0].set_title(f"pass_forw[0] (t=0)")

im11 = axs[1,1].imshow(pf_l, extent=extent)
axs[1,1].set_title(f"pass_forw[-1] (t={time_f:.3f})")

axs[1,2].axis('off')

im03 = axs[1,3].imshow(af_f, extent=extent)
axs[1,3].set_title(f"act_forw[0] (t=0)")

im14 = axs[1,4].imshow(af_l, extent=extent)
axs[1,4].set_title(f"act_forw[-1] (t={time_f:.3f})")

im20 = axs[2,0].imshow(pr_f, extent=extent)
axs[2,0].set_title(f"pass_rev[0] (t=0)")

im21 = axs[2,1].imshow(pr_l, extent=extent)
axs[2,1].set_title(f"pass_rev[-1] (t={time_r:.3f})")

axs[2,2].axis("off")

im23 = axs[2,3].imshow(ar_f, extent=extent)
axs[2,3].set_title(f"act_rev[0] (t=0)")

im24 = axs[2,4].imshow(ar_l, extent=extent)
axs[2,4].set_title(f"act_rev[-1] (t={time_r:.3f})")

if isinstance(addl_time_to_plot, str) and len(addl_time_to_plot) > 0:
    axs[3,0].axis("off")
    
    im31 = axs[3,1].imshow(pr_c, extent=extent)
    axs[3,1].set_title(f"pass_rev[-1] (t={time_c:.3f})")

    axs[3,2].axis("off")
    axs[3,3].axis("off")

    im34 = axs[3,4].imshow(ar_c, extent=extent)
    axs[3,4].set_title(f"act_rev[-1] (t={time_c:.3f})")

fig.tight_layout()
fig.savefig("diffusion.png")
