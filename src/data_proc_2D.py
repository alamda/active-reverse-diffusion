import numpy as np
import scipy.special as special


class DataProc2D:
    def __init__(self, xmin=-2, xmax=2,
                 ymin=-2, ymax=2,
                 num_hist_bins=50):
        self.xmin = xmin
        self.xmax = xmax
        
        self.ymin = ymin
        self.ymax = ymax
        
        self.num_hist_bins = num_hist_bins
        
        self.t_list = None
        self.diff_list = None
        
        
    def calc_KL_divergence(self, target_sample, test_sample):
        
        target_sample_x = np.array(target_sample)[:,0]
        target_sample_y = np.array(target_sample)[:,1]
        
        h_target, xbins_target, ybins_target = \
            np.histogram2d(target_sample_x, 
                           target_sample_y,
                           bins=self.num_hist_bins,
                           density=True)
            
        h_target /= h_target.sum().sum()
        
        test_sample_x = np.array(test_sample)[:,0]
        test_sample_y = np.array(test_sample)[:,1]
            
        h_test, xbins_test, ybins_test = \
            np.histogram2d(test_sample_x,
                           test_sample_y,
                           bins=self.num_hist_bins,
                           density=True)
            
        h_test /= h_test.sum().sum()
        
        breakpoint()
        
        diff = None
        
        try:
            rel_entr = special.rel_entr(h_test, h_target)
            
            if np.isfinite(rel_entr).all():
                diff = np.sum(rel_entr)
            else:
                raise TypeError
        except TypeError:
            print("One or more values in KL divergence is infinite."
                  "Adjust histogram range.")
            
        
        return diff
            
        