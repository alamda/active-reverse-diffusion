from .context import diffusion_numeric, data_proc, noise

from diffusion_numeric import DiffusionNumeric
from data_proc import DataProc
from noise import NoiseActive, NoisePassive

import math
import numpy as np
import torch

class DummyTarget:
    shape = None
    
    def gen_sample(self, shape=None):
        self.shape = shape
        
        self.sample = torch.randn(*self.shape).type(torch.DoubleTensor)  
        
        return self.sample
    
class DiffusionNumericTest_Factory:
    ofile_base = "test"
    
    sample_size = 1000
    
    passive_noise = NoisePassive(T=1.0, dim=sample_size)
    active_noise = NoiseActive(Tp=0, Ta=1.0, tau=0.1, dim=sample_size)
    dummy_target_1D = DummyTarget()
    dummy_target_1D.gen_sample(shape=(sample_size,1))
    dummy_target_2D = DummyTarget()
    dummy_target_2D.gen_sample(shape=(sample_size,2))

    num_diffusion_steps = 10
    dt = 0.5
    k = 2

    data_proc = DataProc()
    
    all_params_explicit_1D = dict(ofile_base=ofile_base,
                                  passive_noise=passive_noise,
                                  active_noise=active_noise,
                                  target=dummy_target_1D,
                                  num_diffusion_steps=num_diffusion_steps,
                                  dt=dt,
                                  k=k,
                                  data_proc=data_proc)
    
    all_params_explicit_2D = dict(ofile_base=ofile_base,
                                  passive_noise=passive_noise,
                                  active_noise=active_noise,
                                  target=dummy_target_2D,                                  
                                  num_diffusion_steps=num_diffusion_steps,
                                  dt=dt,
                                  k=k,
                                  data_proc=data_proc)
    
    init_param_dict = dict(all_params_1D=all_params_explicit_1D,
                           all_params_2D=all_params_explicit_2D)
    
def test_init():
    # Init function basically the same as super()
    pass

# TODO: figure out a way to test this function alone
def test_compute_loss_passive():
    dn_factory = DiffusionNumericTest_Factory()
    
    for _, params in dn_factory.init_param_dict.items():
        dn = DiffusionNumeric(**params)
        
        # TODO

def test_train_diffusion_passive():
    dn_factory = DiffusionNumericTest_Factory()
    
    for _, params in dn_factory.init_param_dict.items():
        dn = DiffusionNumeric(**params)
        
        all_models = dn.train_diffusion_passive(iterations=10)
        
        assert dn.passive_models is not None
        assert len(all_models) == dn_factory.num_diffusion_steps - 1
        

# TODO: finish writing assertions
def test_sample_from_diffusion_passive():
    dn_factory = DiffusionNumericTest_Factory()
    
    for _, params in dn_factory.init_param_dict.items():
        dn = DiffusionNumeric(**params)
        
        dn.train_diffusion_passive(iterations=10)
        
        dn.sample_from_diffusion_passive()
        
        # TODO
        
        # assert len(dn.passive_reverse_time_arr) == ???
        # assert len(dn.passive_reverse_samples) == ???

def test_M_11_12_22():
    dn_factory = DiffusionNumericTest_Factory()
    
    for _, params in dn_factory.init_param_dict.items():
        dn = DiffusionNumeric(**params)
        
        M11, M12, M22 = dn.M_11_12_22(5)
        
        for val in [M11, M12, M22]:
            assert math.isfinite(val)

# TODO: figure out a way to test this function alone      
def test_compute_loss_active():
    pass 

# TODO: finish writing assertions
def test_train_diffusion_active():
    dn_factory = DiffusionNumericTest_Factory()
    
    for _, params in dn_factory.init_param_dict.items():
        dn = DiffusionNumeric(**params)
        
        all_models = dn.train_diffusion_active(iterations=10)
        
        assert dn.active_models_x is not None
        assert dn.active_models_eta is not None
        
        # TODO
        
        # assert len(all_models) == dn_factory.num_diffusion_steps - 1

# TODO: finish writing assertions       
def test_sample_from_diffusion_active():
    dn_factory = DiffusionNumericTest_Factory()
    
    for _, params in dn_factory.init_param_dict.items():
        dn = DiffusionNumeric(**params)
        
        all_models_x, all_models_eta = dn.train_diffusion_active(iterations=10)

        dn.sample_from_diffusion_active()
        
        # TODO