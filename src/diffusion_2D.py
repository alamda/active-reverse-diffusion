from abc import ABC as AbstractBaseClass
from abc import abstractmethod

import multiprocessing
from multiprocess import Pool

import numpy as np
import torch


class Diffusion2D(AbstractBaseClass):
    def __init__(self, ofile_base="", passive_noise=None, active_noise=None, 
                 num_diffusion_steps=None, dt=None, k=1, sample_dim=None, 
                 data_proc=None, diffusion_type=None):
        
        self.ofile_base = ofile_base
        
        self.passive_noise = passive_noise
        self.active_noise = active_noise
        
        self.target = target
        
        self.num_diffusion_steps = num_diffusion_steps
        self.dt = dt
        
        self.k = k
        
        self.sample_dim = sample_dim
        
        self.data_proc = data_proc
        
        self.passive_forward_time_arr = None
        self.passive_forward_samples = None
        self.passive_reverse_time_arr = None
        self.passive_reverse_samples = None
        self.passive_diff_list = None

        self.active_forward_time_arr = None
        self.active_forward_samples_x = None
        self.active_forward_samples_eta = None
        self.active_reverse_time_arr = None
        self.active_reverse_samples_x = None
        self.active_reverse_samples_eta = None
        self.active_diff_list = None

        self.diffusion_type = diffusion_type

        self.num_passive_reverse_difussion_steps = None
        self.num_active_reverse_diffusion_steps = None
        
    def forward_diffusion_passive(self):
        forward_diffusion_sample_list = [self.target.sample]
        
        