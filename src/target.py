from abc import ABC as AbstractBaseClass
from abc import abstractmethod

from data_handler import DiffusionSampleHandler

import matplotlib.pyplot as plt
import pickle
import numpy as np
import mmap
import torch
import os
from tqdm.auto import tqdm 
import gc

class TargetAbstract(AbstractBaseClass):
    """Abstract class for the target distribution"""

    def __init__(self, name="target", 
                 sample_size=None, 
                 sample_dim=None,
                 xmin=None, 
                 xmax=None,
                 ymin=None,
                 ymax=None, 
                 target_sample_fname="target.npy"):
        
        self.name = name
        self.sample_size = int(sample_size)
        self.sample_dim = int(sample_dim)

        self.xmin = xmin
        self.xmax = xmax
        
        self.ymin = ymin if ymin is not None else self.xmin
        self.ymax = ymax if ymax is not None else self.xmax

        self.sample = None

        self.x_arr = None
        self.y_arr = None
        self.prob_arr = None
        
        self.target_hist_idx_prob_arr = None
        self.target_hist_idx_arr = None
        
        self.target_sample_fname = target_sample_fname
        self.target_sample_data_h = DiffusionSampleHandler(fname=self.target_sample_fname,
                                                        sample_size=self.sample_size,
                                                        sample_dim=self.sample_dim)
    
    @abstractmethod
    def gen_target_sample(self):
        """Define the target dsn and sample it after it was initialized"""

    def gen_target_sample_to_file(self):
        self.target_sample_data_h.create_new_file()
            
        with open(self.target_sample_fname, 'wb') as f:
            gen_batch_size = 10000
            
            num_batches = self.sample_size // gen_batch_size
            
            sample_mod_batch = self.sample_size % gen_batch_size
            
            bar = tqdm(range(num_batches))
            
            bar.set_description("Target sample generation")
            
            def gen_samples(num_samples):
                sample_idx_list = np.random.choice(len(self.target_hist_idx_arr), num_samples, 
                                              p=self.target_hist_idx_prob_arr)

                # Only implemented for 2D for now
                x_samples = [self.x_arr[self.target_hist_idx_arr[idx][0]] for idx in sample_idx_list]
                y_samples = [self.y_arr[self.target_hist_idx_arr[idx][1]] for idx in sample_idx_list]        
                
                return torch.from_numpy(np.array(list(zip(x_samples, y_samples))))
            
            for e in bar:
                sample = gen_samples(gen_batch_size)
                 
                self.target_sample_data_h.write_tensor_to_file(tensor=sample)      
                
                del sample
                gc.collect()
            
            if sample_mod_batch > 0: 
                sample = gen_samples(sample_mod_batch)
                
                self.target_sample_data_h.write_tensor_to_file(tensor=sample)            
                                
                del sample
                gc.collect()
    
    def mmap_sample(self):
        self.sample = self.target_sample_data_h.mmap_tensor_from_file()
    
    def close_mmap(self):
        self.target_sample_data_h.close_mmap()
      
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
