from abc import ABC as AbstractBaseClass
from abc import abstractmethod

import multiprocess
from multiprocess import Pool

import numpy as np
import torch


class Diffusion2D(AbstractBaseClass):
    def __init__(self, ofile_base="", passive_noise=None, active_noise=None, 
                 target=None, num_diffusion_steps=None, dt=None, k=1, 
                 sample_dim=None, data_proc=None, diffusion_type=None):
        
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
        
        sample_t = self.target.sample
        
        sample_shape = sample_t.shape

        for t_idx in range(self.num_diffusion_steps):
            sample_t = sample_t - self.dt*sample_t + \
                    np.sqrt(2*self.passive_noise.temperature*self.dt) * \
                    np.random.randn(sample_shape[0], sample_shape[1])

            forward_diffusion_sample_list.append(sample_t)
        
       
        self.passive_forward_samples = forward_diffusion_sample_list
        
        return self.passive_forward_samples
    
    @abstractmethod
    def sample_from_diffusion_passive(self):
        """Reverse diffusion process with passive noise"""

    def calculate_passive_diff_list(self, multiproc=True):
        if self.data_proc is not None:
            if multiproc == True:

                num_cpus = multiprocess.cpu_count()
                num_procs = num_cpus - 4

                with Pool() as pool:
                    self.passive_diff_list = \
                        self.data_proc.calc_diff_vs_t_multiproc(self.target.sample,
                                                                self.passive_reverse_samples,
                                                                pool=pool)
            else:
                self.passive_diff_list = self.data_proc.calc_diff_vs_t(self.target.sample,
                                                                       self.passive_reverse_samples)