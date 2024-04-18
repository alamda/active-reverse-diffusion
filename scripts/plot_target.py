import pickle 
import numpy as np
from matplotlib import figure

with open("target_sample.pkl", "rb") as f:
    target_sample = pickle.load(f)

sample_x = target_sample[:,0].numpy()
sample_y = target_sample[:,1].numpy()

fig = figure.Figure()
ax = fig.subplots(1)

hist, _, _ = np.histogram2d(sample_x, sample_y)

im = ax.imshow(hist)
fig.colorbar(im)

fig.savefig("target_sample.png")
