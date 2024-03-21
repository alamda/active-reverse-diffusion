from abc import ABC as AbstractBaseClass
from abc import abstractmethod

import multiprocess
from multiprocess import Pool


class Diffusion(AbstractBaseClass):
    def __init__(self, ofile_base="", passive_noise=None, active_noise=None, target=None,
                 num_diffusion_steps=None, dt=None, k=1, sample_dim=None, data_proc=None):
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

    @abstractmethod
    def forward_diffusion_passive(self):
        """Forward diffusion process with passive noise"""

    @abstractmethod
    def sample_from_diffusion_passive(self):
        """Reverse diffusion process with passive noise"""

    @abstractmethod
    def forward_diffusion_active(self):
        """Forward diffusion process with active and passive noise"""

    @abstractmethod
    def sample_from_diffusion_active(self):
        """Reverse diffusion process with active and passive noise"""

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

    def calculate_active_diff_list(self, multiproc=True):
        if self.data_proc is not None:
            if multiproc == True:

                num_cpus = multiprocess.cpu_count()
                num_procs = num_cpus - 4

                with Pool(processes=num_procs) as pool:
                    self.active_diff_list = \
                        self.data_proc.calc_diff_vs_t_multiproc(self.target.sample,
                                                                self.active_reverse_samples_x,
                                                                pool=pool)
            else:
                self.active_diff_list = self.data_proc.calc_diff_vs_t(self.target.sample,
                                                                      self.active_reverse_samples_x)
