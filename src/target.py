from abc import ABC as AbstractBaseClass
from abc import abstractmethod

import matplotlib.pyplot as plt
import pickle
import numpy as np


class TargetAbstract(AbstractBaseClass):
    """Abstract class for the target distribution"""

    def __init__(self, name="target", 
                 sample_size=None, 
                 sample_dim=None,
                 xmin=None, 
                 xmax=None,
                 ymin=None,
                 ymax=None):
        
        self.name = name
        self.sample_size = sample_size
        self.sample_dim = sample_dim

        self.xmin = xmin
        self.xmax = xmax
        
        self.ymin = ymin if ymin is not None else self.xmin
        self.ymax = ymax if ymax is not None else self.xmax

        self.sample = None

        self.x_arr = None
        self.y_arr = None
        self.prob_arr = None
        
    @abstractmethod
    def gen_target_sample(self):
        """Define the target dsn and sample it after it was initialized"""

    def plot_target_hist(self,
                        fname="target.png",
                        title="example target sample",
                        num_bins=100,
                        num_x_bins=None,
                        num_y_bins=None,
                        hist_range=None,
                        hist_x_range=None,
                        hist_y_range=None):
        
        num_x_bins = num_x_bins \
            if num_x_bins is not None \
            else num_bins
            
        num_y_bins = num_y_bins \
            if num_y_bins is not None \
            else num_x_bins
        
        hist_x_range = hist_x_range \
            if hist_x_range is not None \
            else hist_range
            
        hist_y_range = hist_y_range \
            if hist_y_range is not None \
            else hist_range
            
        hist_x_range = hist_x_range \
            if hist_x_range is not None \
            else (self.xmin, self.xmax)
            
        hist_y_range = hist_y_range \
            if hist_y_range is not None \
            else (self.ymin, self.ymax)
        
        if self.sample_dim is None:
            self.sample_dim = self.sample.shape[1]
        
        fig, ax = plt.subplots()
        
        if self.sample_dim == 1:
            
            hist, x_bins, _ = ax.hist(self.sample,
                                      density=True,
                                      bins=num_x_bins,
                                      range=[hist_x_range[0], hist_x_range[1]])
            
        elif self.sample_dim == 2:
            hist, x_bins, y_bins = np.histogram2d(self.sample[:,0],
                                                  self.sample[:,1],
                                                  density=True,
                                                  bins=[num_x_bins, num_y_bins],
                                                  range=[[hist_x_range[0], hist_x_range[1]],
                                                         [hist_y_range[0], hist_y_range[1]]])
        
            ax.imshow(hist)
        
        ax.set_title(title)
        
        plt.savefig(fname)
        
        plt.close(fig)