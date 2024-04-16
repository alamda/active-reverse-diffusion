from abc import ABC as AbstractBaseClass
from abc import abstractmethod

import multiprocess
from multiprocess import Pool

import numpy as np
import torch
import mmap

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
                 sample_size=None, 
                 forward_passive_fname="forward_passive.npy",
                 forward_x_active_fname="forward_x_active.npy", 
                 forward_eta_active_fname="forward_eta_active.npy",
                 reverse_passive_fname="reverse_passive.npy", 
                 reverse_x_active_fname="reverse_x_active.npy", 
                 reverse_eta_active_fname="reverse_eta_active.npy"):
        
        self.ofile_base = str(ofile_base)
        self.forward_passive_fname = forward_passive_fname
        self.forward_x_active_fname = forward_x_active_fname
        self.forward_eta_active_fname = forward_eta_active_fname
        self.reverse_passive_fname = reverse_passive_fname
        self.reverse_x_active_fname = reverse_x_active_fname
        self.reverse_eta_active_fname = reverse_eta_active_fname

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
        with open(self.forward_passive_fname, "wb") as f:
            f.write(self.target.sample.numpy().tobytes())
        
        sample_t = self.target.sample.detach()
        
        for t_idx in range(self.num_diffusion_steps):
            sample_t = sample_t - self.dt*sample_t + \
                np.sqrt(2*self.passive_noise.temperature*self.dt) * \
                torch.normal(torch.zeros_like(self.target.sample),
                             torch.ones_like(self.target.sample))
            
            with open(self.forward_passive_fname, "ab") as f:
                f.write(sample_t.numpy().tobytes())
       
        with open(self.forward_passive_fname, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            mm_arr = np.frombuffer(mm, dtype=np.double)
            
            _ = mm_arr.shape
            
            self.passive_forward_samples = \
                torch.from_numpy(np.reshape(mm_arr, \
                    (self.num_diffusion_steps + 1, self.sample_size, self.sample_dim)))
        
            self.passive_forward_samples = [ x for x in self.passive_forward_samples]
        
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
        
        with open(self.forward_x_active_fname, "wb") as f:
            f.write(self.target.sample.numpy().tobytes())
        
        with open(self.forward_eta_active_fname, "wb") as f:
            f.write(eta.numpy().tobytes())
        
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

            with open(self.forward_x_active_fname, "ab") as f:
                f.write(sample_t.numpy().tobytes())
                
            with open(self.forward_eta_active_fname, "ab") as f:
                f.write(eta.numpy().tobytes())
                
        with open(self.forward_x_active_fname, "r+b") as f:
            mm_x = mmap.mmap(f.fileno(), 0)
            mm_x_arr = np.frombuffer(mm_x, dtype=np.double)
            
            _ = mm_x_arr.shape
            
            self.active_forward_samples_x = \
                torch.from_numpy(np.reshape(mm_x_arr, 
                                            (self.num_diffusion_steps + 1, self.sample_size, self.sample_dim))) 
            
            self.active_forward_samples_x = [ x for x in self.active_forward_samples_x]    
        
        with open(self.forward_eta_active_fname, "r+b") as f:
            mm_eta = mmap.mmap(f.fileno(), 0)
            mm_eta_arr = np.frombuffer(mm_eta, dtype=np.double)
            
            _ = mm_eta_arr.shape
            
            self.active_forward_samples_eta = \
                torch.from_numpy(np.reshape(mm_eta_arr, 
                                            (self.num_diffusion_steps + 1, self.sample_size, self.sample_dim))) 

            self.active_forward_samples_eta = [eta for eta in self.active_forward_samples_eta]

        return self.active_forward_samples_x, self.active_forward_samples_eta

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