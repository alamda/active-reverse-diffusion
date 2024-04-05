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
    sample = torch.randn(*shape).type(torch.DoubleTensor)

class DiffusionTest_Factory:    
    ofile_base = "test"
    
    sample_size = 1000
    sample_dim = 2
    
    passive_noise = NoisePassive(T=1.0, dim=sample_size)
    active_noise = NoiseActive(Tp=0, Ta=1.0, tau=0.1, dim=sample_size)
    dummy_target = DummyTarget()
    
    num_diffusion_steps = 10
    dt = 0.5
    k = 2

    data_proc = DataProc()
    
    diffusion_type = 'numeric'

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
    
    all_params_explicit = dict(ofile_base=ofile_base,
                               sample_size=sample_size,
                               sample_dim=sample_dim,
                               passive_noise=passive_noise,
                               active_noise=active_noise,
                               target=dummy_target,
                               num_diffusion_steps=num_diffusion_steps,
                               dt=dt,
                               k=k,
                               data_proc=data_proc,
                               diffusion_type=diffusion_type)
    
    req_params_explicit = dict(passive_noise=passive_noise,
                               active_noise=active_noise,
                               target=dummy_target,
                               num_diffusion_steps=num_diffusion_steps,
                               dt=dt,
                               data_proc=data_proc,
                               diffusion_type=diffusion_type)
    
    init_param_dict = dict(all_params=all_params_explicit,
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
    
    dn = Diffusion(**dn_factory.all_params_explicit)
    
    passive_forward_samples = dn.forward_diffusion_passive()
    
    assert len(passive_forward_samples) == dn_factory.num_diffusion_steps + 1
    
    for sample in passive_forward_samples: 
        assert bool(torch.isfinite(sample).all())
    