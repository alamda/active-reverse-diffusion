from .context import data_proc

from data_proc import DataProc

import math
import numpy as np
import multiprocess
from multiprocess import Pool

class DataProcTest_Factory:
    var_dict = dict(xmin=float, 
                    xmax=float,
                    ymin=float, 
                    ymax=float,
                    num_hist_bins=(list, tuple),
                    num_hist_bins_x=int,
                    num_hist_bins_y=int)
    
    ## Params for __init__()
    # All params explicitly specified
    init_all_params = dict(xmin=-5, 
                  xmax=-1, 
                  ymin=-2, 
                  ymax=0,
                  num_hist_bins=(4,4),
                  num_hist_bins_x=4,
                  num_hist_bins_y=6)
    
    # Only params with default values specified
    init_preset_params = dict(xmin=-5, 
                         xmax=-1, 
                         num_hist_bins=(4,4))
    
    # Only non-preset params
    init_non_preset_params = dict(ymin=1,
                             ymax=10,
                             num_hist_bins=(4,4))
    
    # No params
    init_no_params = dict()

    init_param_dict = dict(all_params=init_all_params,
                           preset_params=init_preset_params,
                           non_preset_params=init_non_preset_params,
                           no_params=init_no_params)
    
    ## Dummy diffusion data    
    dummy_sample_shape = (1000, 2)
    dummy_target_sample = np.random.rand(*dummy_sample_shape)
    
    dummy_diffusion_sample_list = []
    dummy_num_diffusion_steps = 10
    
    for _ in range(dummy_num_diffusion_steps):
        dummy_diffusion_sample_list.append(np.random.rand(*dummy_sample_shape))
    
    ## Params for calc_diff_vs_t
    num_diffusion_steps = 10
    
    num_procs = multiprocess.cpu_count() - 2
    
    no_multiproc = dict()
    
    pool = Pool(processes=num_procs)
     
    multiproc = dict(multiproc=True,
                     pool=pool)
    
    calc_param_dict = dict(no_multiproc=no_multiproc,
                           multiproc=multiproc)

def test_init():
    
    dp_factory = DataProcTest_Factory()
    
    for _, params in dp_factory.init_param_dict.items():
        dp = DataProc(**params)
    
        for val_name, val_type in dp_factory.var_dict.items():
            val = getattr(dp, val_name)
            
            assert isinstance(val, val_type)
            
            if (type(val) == float) or (type(val) == int):
                assert math.isfinite(val)
        
        # Check that histogram dimension tuple is ok
        for num in dp.num_hist_bins:
            assert isinstance(num, int)
            assert math.isfinite(num)
               
def test_calc_KL_divergence():
    
    dp_factory = DataProcTest_Factory()
    
    sample_shape_list = [(1000,),
                         (1000,1),
                         (1000,2)]
    
    dp = DataProc(xmin=0.1, xmax=0.9, num_hist_bins=5)
    
    for shape in sample_shape_list:
        target_sample = np.random.rand(*shape)
        test_sample = np.random.rand(*shape)
        
        diff = dp.calc_KL_divergence(target_sample=target_sample,
                                     test_sample=test_sample)

        assert diff is not None
        assert isinstance(diff, float)
        assert math.isfinite(diff)
        
def test_calc_diff_vs_t():
    
    dp_factory = DataProcTest_Factory()
    
    dp = DataProc(xmin=0.1, xmax=0.9, num_hist_bins=5)
  
    for _, params in dp_factory.calc_param_dict.items():
        diff_list = dp.calc_diff_vs_t(target_sample=dp_factory.dummy_target_sample, 
                                    diffusion_sample_list=dp_factory.dummy_diffusion_sample_list)
        
        assert len(diff_list) == dp_factory.dummy_num_diffusion_steps - 1 #????
        assert math.isfinite(np.array(diff_list).all())
        
def test_calc_diff_vs_t_multiproc():
    dp_factory = DataProcTest_Factory()
    
    dp = DataProc(xmin=0.1, xmax=0.9, num_hist_bins=5)
    
    diff_list = dp.calc_diff_vs_t_multiproc(target_sample=dp_factory.dummy_target_sample,
                                            diffusion_sample_list=dp_factory.dummy_diffusion_sample_list,
                                            pool=dp_factory.pool)
    
    assert len(diff_list) == dp_factory.dummy_num_diffusion_steps - 1
    assert math.isfinite(np.array(diff_list).all())
