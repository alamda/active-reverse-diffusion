from .context import target

from target import Target

import math
import numpy as np
import torch

Target.__abstractmethods__ = set()

class TargetTest_Factory:
    var_dict = dict(name=str,
                    sample_size=int,
                    sample_dim=int,
                    xmin=float,
                    xmax=float,
                    ymin=float,
                    ymax=float)
    
    name = "target test"
    
    sample_size = 1000
    
    xmin=-5
    xmax=5
    ymin=-4
    ymax=4
    
    init_all_params_1D = dict(name=name,
                              sample_size=sample_size,
                              sample_dim=1,
                              xmin=xmin,
                              xmax=xmax,
                              ymin=ymin,
                              ymax=ymax)
    
    init_all_params_2D = dict(name=name,
                              sample_size=sample_size,
                              sample_dim=2,
                              xmin=xmin,
                              xmax=xmax,
                              ymin=ymin,
                              ymax=ymax)
    
    init_partial_params_2D = dict(sample_size=sample_size,
                                  sample_dim=2,
                                  xmin=-10,
                                  xmax=10)
    
    init_param_dict = dict(all_params_1D=init_all_params_1D,
                           partial_params_2D=init_partial_params_2D)
    
def test_init():
    tg_factory = TargetTest_Factory()
    
    for _, params in tg_factory.init_param_dict.items():
        tg = Target(**params)
        
        for val_name, val_type in tg_factory.var_dict.items():
            val = getattr(tg, val_name)
            
            assert isinstance(val, val_type)
            
            if (type(val) == float) or (type(val) == int):
                assert math.isfinite(val)
                
            
    
    
    
    

