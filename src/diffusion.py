from abc import ABC as AbstractBaseClass
from abc import abstractmethod

import multiprocess
from multiprocess import Pool

import numpy as np
import torch

class Diffusion(AbstractBaseClass):
    def __init__(self, ofile_base="", 
                 passive_noise=None, 
                 active_noise=None, 
                 target=None,
                 num_diffusion_steps=None, 
                 dt=None, 
                 k=1, 
                 data_proc=None, 
                 diffusion_type=None,
                 sample_size=None):
        
        self.ofile_base = str(ofile_base)

        self.passive_noise = passive_noise
        self.active_noise = active_noise

        self.target = target

        self.num_diffusion_steps = int(num_diffusion_steps)
        self.dt = float(dt)

        self.k = float(k)
        
        self.sample_size = sample_size

        # Number of samples generated - inferred from target sample size
        self.sample_size = int(self.target.sample.shape[0])
        
        if len(self.target.sample.shape) == 1:
            self.sample_dim = 1
        else:
            # 1D vs 2D - inferred from target dimension
            self.sample_dim = int(self.target.sample.shape[1])

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

        self.diffusion_type = str(diffusion_type)

        self.num_passive_reverse_difussion_steps = None
        self.num_active_reverse_diffusion_steps = None

    def forward_diffusion_passive(self):
        forward_diffusion_sample_list = [self.target.sample]
        
        sample_t = self.target.sample.detach()
        
        for t_idx in range(self.num_diffusion_steps):
            sample_t = sample_t - self.dt*sample_t + \
                np.sqrt(2*self.passive_noise.temperature*self.dt) * \
                torch.normal(torch.zeros_like(self.target.sample),
                             torch.ones_like(self.target.sample))

            forward_diffusion_sample_list.append(sample_t)
       
        self.passive_forward_samples = forward_diffusion_sample_list
        
        return self.passive_forward_samples

    @abstractmethod
    def sample_from_diffusion_passive(self):
        """Reverse diffusion process with passive noise"""

    def forward_diffusion_active(self):
        eta = torch.normal(torch.zeros_like(self.target.sample),
                           np.sqrt(self.active_noise.temperature.active /
                                   self.active_noise.correlation_time)
                           * torch.ones_like(self.target.sample)
                           )
        samples = [self.target.sample]
        eta_samples = [eta]
        sample_t = self.target.sample

        for t_idx in range(self.num_diffusion_steps):
            sample_t = sample_t - self.dt*sample_t + self.dt*eta + \
                np.sqrt(2*self.active_noise.temperature.passive*self.dt) * \
                torch.normal(torch.zeros_like(self.target.sample),
                             torch.ones_like(self.target.sample))

            eta = eta - (1/self.active_noise.correlation_time)*self.dt*eta + \
                (1/self.active_noise.correlation_time) * \
                np.sqrt(2*self.active_noise.temperature.active*self.dt) * \
                torch.normal(torch.zeros_like(eta), torch.ones_like(eta))

            samples.append(sample_t)
            eta_samples.append(eta)

        samples = [s.reshape((self.sample_size, self.sample_dim)).type(torch.DoubleTensor)
                   for s in samples]

        eta_samples = [s.reshape((self.sample_size, self.sample_dim)).type(torch.DoubleTensor)
                       for s in eta_samples]

        self.active_forward_samples_x = samples
        self.active_forward_samples_eta = eta_samples

        return samples, eta_samples

    @abstractmethod
    def sample_from_diffusion_active(self):
        """Reverse diffusion process with active and passive noise"""

    def calculate_diff_list(self, diffusion_type=None, multiproc=True):
        sample_list_attr_name = None
        diff_list_attr_name = None
        
        if diffusion_type in ('passive', 'Passive', 'PASSIVE'):
            sample_list_attr_name = 'passive_reverse_samples'
            diff_list_attr_name = 'passive_diff_list'
        elif diffusion_type in ('active', 'Active', 'ACTIVE'):
            sample_list_attr_name = 'active_reverse_samples_x'
            diff_list_attr_name = 'active_diff_list'
        
        if (sample_list_attr_name is not None) and (diff_list_attr_name is not None):
            if self.data_proc is not None:
                sample_list = getattr(self, sample_list_attr_name)
                
                if multiproc == True:

                    num_cpus = multiprocess.cpu_count()
                    num_procs = num_cpus - 4

                    with Pool() as pool:
                        diff_list = \
                            self.data_proc.calc_diff_vs_t(target_sample=self.target.sample,
                                                          diffusion_sample_list=sample_list,
                                                          multiproc=True,
                                                          pool=pool)
                        
                        setattr(self, diff_list_attr_name, diff_list)
                else:
                    diff_list = self.data_proc.calc_diff_vs_t(self.target.sample,
                                                              diffusion_sample_list=sample_list,
                                                              multiproc=False,
                                                              pool=None)
                    
                    setattr(self, diff_list_attr_name, diff_list)

        else:
            print("Invalied diffusion type (use either 'passive' or 'active')")

    def calculate_passive_diff_list(self, multiproc=True):
        self.calculate_diff_list(diffusion_type='passive', multiproc=multiproc)
        
    def calculate_active_diff_list(self, multiproc=True):
        self.calculate_diff_list(diffusion_type='active', multiproc=multiproc)