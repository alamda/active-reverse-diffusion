import numpy as np
import tqdm
import scipy.special as special

class DataProc:
    def __init__(self, xmin=-10, xmax=-10, 
                 ymin=None, ymax=None, 
                 num_hist_bins=(10,10), 
                 num_hist_bins_x=None,
                 num_hist_bins_y=None):
        
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        
        self.ymin = float(ymin) if ymin is not None else float(xmin)
        self.ymax = float(ymax) if ymax is not None else float(xmax)
        
        if isinstance(num_hist_bins, int):
            self.num_hist_bins = (num_hist_bins, num_hist_bins)
        elif isinstance(num_hist_bins, tuple):
            self.num_hist_bins = num_hist_bins
        
        self.num_hist_bins_x = int(num_hist_bins_x) \
            if num_hist_bins_x is not None \
            else int(self.num_hist_bins[0])
            
        self.num_hist_bins_y = int(num_hist_bins_y) \
            if num_hist_bins_y is not None \
            else int(self.num_hist_bins[1])
            
        if num_hist_bins_x is not None:
            self.num_hist_bins = (num_hist_bins_x, self.num_hist_bins[1])
        if num_hist_bins_y is not None:
            self.num_hist_bins = (self.num_hist_bins[0], num_hist_bins_y)
            
        self.t_list = None
        self.diff_list = None
        
    def calc_KL_divergence(self, target_sample=None, test_sample=None):
        try:
            if not target_sample.shape == test_sample.shape:
                raise AssertionError
        except AssertionError:
            print("target and test samples must have the same shape")
              
        target_sample_size = target_sample.shape[0]
        
        if len(target_sample.shape) == 1:
            target_sample_dim = 1
        else:
            target_sample_dim = target_sample.shape[1]
        
        test_sample_size = test_sample.shape[0]
        
        if len(test_sample.shape) == 1:
            test_sample_dim = 1
        else:
            test_sample_dim = test_sample.shape[1]
        
        if (target_sample_dim == 1) and (test_sample_dim == 1):
            h_target, _ = np.histogram(target_sample,
                                       bins=self.num_hist_bins[0],
                                       density=True,
                                       range=(self.xmin, self.xmax))
            
            h_target /= h_target.sum()
            
            h_test, _ = np.histogram(test_sample,
                                     bins=self.num_hist_bins[0],
                                     density=True,
                                     range=(self.xmin, self.xmax))

            h_test /= h_test.sum()
            
        elif (target_sample_dim == 2) and (test_sample_dim == 2):
            target_sample_x = np.array(target_sample)[:,0]
            target_sample_y = np.array(target_sample)[:,1]
            
            h_target, _, _ =np.histogram2d(target_sample_x, 
                                           target_sample_y,
                                           bins=[self.num_hist_bins_x,
                                                 self.num_hist_bins_y],
                                           density=True,
                                           range=[[self.xmin, self.xmax],
                                                  [self.ymin, self.ymax]])

            h_target /= h_target.sum().sum()

            test_sample_x = np.array(test_sample)[:,0]
            test_sample_y = np.array(test_sample)[:,1]
                
            h_test, _, _ = np.histogram2d(test_sample_x,
                                          test_sample_y,
                                          bins=[self.num_hist_bins_x,
                                                self.num_hist_bins_y],
                                          density=True,
                                          range=[[self.xmin, self.xmax],
                                                 [self.ymin, self.ymax]])

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

            for t_idx in range(num_diffusion_steps-1):

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
                for t_idx in range(num_diffusion_steps-1):
                    diff = self.calc_KL_divergence(target_sample,
                                                   diffusion_sample_list[t_idx])

                    t_list.append(t_idx)
                    diff_list.append(diff)
                    pbar.update()

        self.t_list = t_list
        self.diff_list = diff_list

        return diff_list

    def calc_diff_vs_t_multiproc(self, target_sample=None, 
                                 diffusion_sample_list=None, 
                                 pool=None):
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
