from .context import diffusion, data_proc, noise

from data_proc import DataProc
from diffusion import Diffusion
from noise import NoiseActive, NoisePassive

import math
import numpy as np
import torch

Diffusion.__abstractmethods__=set()

class DummyTarget:
    shape = (1000, 2)
    
    def gen_sample(self, shape=None):
        self.shape = shape if shape is not None else self.shape
        
        self.sample = torch.randn(*self.shape).type(torch.DoubleTensor)  
        
        return self.sample

class DiffusionTest_Factory:    
    ofile_base = "test"
    
    sample_size = 1000
    
    passive_noise = NoisePassive(T=1.0, dim=sample_size)
    active_noise = NoiseActive(Tp=0, Ta=1.0, tau=0.1, dim=sample_size)
    dummy_target_1D = DummyTarget()
    dummy_target_1D.gen_sample(shape=(1000,1))
    dummy_target_2D = DummyTarget()
    dummy_target_2D.gen_sample(shape=(1000,2))

    num_diffusion_steps = 10
    dt = 0.5
    k = 2

    data_proc = DataProc()
    
    diffusion_type = 'numeric'
    
    # Dummy reverse samples because reverse diffusion not impemented in this abstract class
    passive_reverse_samples_1D = []
    passive_reverse_samples_2D = []
    active_reverse_samples_1D = []
    active_reverse_samples_2D = []
    
    for _ in range(num_diffusion_steps):
        passive_reverse_samples_1D.append(dummy_target_1D.gen_sample())
        passive_reverse_samples_2D.append(dummy_target_2D.gen_sample())
        active_reverse_samples_1D.append(dummy_target_1D.gen_sample())
        active_reverse_samples_2D.append(dummy_target_2D.gen_sample())

    ## Testing various sets of parameters for init
    var_dict = dict(ofile_base=str,
                    sample_size=int,
                    sample_dim=int,
                    passive_noise=NoisePassive,
                    active_noise=NoiseActive,
                    target=DummyTarget,
                    num_diffusion_steps=int,
                    dt=float,
                    k=float,
                    data_proc=DataProc,
                    diffusion_type=str)
    
    all_params_explicit_1D = dict(ofile_base=ofile_base,
                                  passive_noise=passive_noise,
                                  active_noise=active_noise,
                                  target=dummy_target_1D,
                                  num_diffusion_steps=num_diffusion_steps,
                                  dt=dt,
                                  k=k,
                                  data_proc=data_proc,
                                  diffusion_type=diffusion_type)
    
    all_params_explicit_2D = dict(ofile_base=ofile_base,
                                  passive_noise=passive_noise,
                                  active_noise=active_noise,
                                  target=dummy_target_2D,                                  
                                  num_diffusion_steps=num_diffusion_steps,
                                  dt=dt,
                                  k=k,
                                  data_proc=data_proc,
                                  diffusion_type=diffusion_type)
    
    req_params_explicit = dict(passive_noise=passive_noise,
                               active_noise=active_noise,
                               target=dummy_target_1D,
                               num_diffusion_steps=num_diffusion_steps,
                               dt=dt,
                               data_proc=data_proc,
                               diffusion_type=diffusion_type) 
    
    init_param_dict = dict(all_params_1D=all_params_explicit_1D,
                           all_params_2D=all_params_explicit_2D,
                           req_params=req_params_explicit)
    

def test_init():
    dn_factory = DiffusionTest_Factory()
    
    for _, params in dn_factory.init_param_dict.items():
        
        dn = Diffusion(**params)
        
        for val_name, val_type in dn_factory.var_dict.items():
            val = getattr(dn, val_name)
            
            assert isinstance(val, val_type)
            
            if (type(val) == float) or (type(val) == int):
                assert math.isfinite(val)
                
def test_forward_diffusion_passive():
    dn_factory = DiffusionTest_Factory()
    
    for _, params in dn_factory.init_param_dict.items():
        dn = Diffusion(**params)
        
        passive_forward_samples = dn.forward_diffusion_passive()
        
        assert len(passive_forward_samples) == dn_factory.num_diffusion_steps + 1
        
        for sample in passive_forward_samples: 
            assert bool(torch.isfinite(sample).all())
            
def test_forward_diffusion_active():
    dn_factory = DiffusionTest_Factory()
    
    for _, params in dn_factory.init_param_dict.items():
        dn = Diffusion(**params)
        
        samples_list = dn.forward_diffusion_active()
        
        for samples in samples_list:
            assert len(samples) == dn_factory.num_diffusion_steps + 1
            
            for sample in samples:
                assert bool(torch.isfinite(sample).all())
                
def test_calculate_diff_list():
    dn_factory = DiffusionTest_Factory()
    
    passive_diff_strings = ['passive', 'Passive', 'PASSIVE']
    active_diff_strings = ['active', 'Active', 'ACTIVE']
    
    for _, params in dn_factory.init_param_dict.items():
        dn = Diffusion(**params)
        
        if dn.sample_dim == 1:
            dn.passive_reverse_samples = dn_factory.passive_reverse_samples_1D
            dn.active_reverse_samples = dn_factory.active_reverse_samples_1D
        elif dn.sample_dim == 2: 
            dn.passive_reverse_samples = dn_factory.passive_reverse_samples_2D
            dn.active_reverse_samples = dn_factory.active_reverse_samples_2D

        for type_str in passive_diff_strings:
            dn.calculate_diff_list(diffusion_type=type_str)
            
            assert dn.passive_diff_list is not None
            
        for type_str in active_diff_strings:
            dn.calculate_diff_list(diffusion_type=type_str)
            
            assert dn.active_diff_list is not None
        