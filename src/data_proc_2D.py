import numpy as np
import scipy.special as special

import tqdm

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
            
    def calc_diff_vs_t(self, target_sample,
                       diffusion_sample_list,
                       multiproc=False,
                       pool=None):
        
        t_list = []
        
        diff_list = []

        num_diffusion_steps = len(diffusion_sample_list)

        proc_list = None

        if multiproc and (pool is not None):
            print("Calculating KL divergences with multiprocessing enabled")

            proc_list = []

            for t_idx in range(num_diffusion_steps - 1):

                proc = pool.apply_async(self.calc_KL_divergence,
                                        (target_sample, diffusion_sample_list[t_idx]))

                proc_list.append(proc)
                t_list.append(t_idx)

            with tqdm.tqdm(total=len(proc_list)) as pbar:
                for proc in proc_list:
                    diff_list.append(proc.get())
                    pbar.update()
        else:
            with tqdm.tqdm(total=num_diffusion_steps) as pbar:
                for t_idx in range(0, num_diffusion_steps-1):
                    diff = self.calc_KL_divergence(target_sample,
                                                   diffusion_sample_list[t_idx])

                    t_list.append(t_idx)
                    diff_list.append(diff)
                    pbar.update()

        self.t_list = t_list
        self.diff_list = diff_list

        return diff_list

    def calc_diff_vs_t_multiproc(self, target_sample, diffusion_sample_list, pool=None):
        if pool is not None:
            print("Calculating KL divergences with multiprocessing enabled")

            diff_list = self.calc_diff_vs_t(target_sample,
                                            diffusion_sample_list,
                                            multiproc=True,
                                            pool=pool)

            return diff_list
        else:
            print(
                "No pool object provided, not using multiprocessing to calculate KL divergences")

            return self.calc_diff_vs_t(target_sample, diffusion_sample_list)
