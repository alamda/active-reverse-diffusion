from target import TargetAbstract

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import gc

import os


class TargetMultiGaussian2D(TargetAbstract):
    def __init__(self, name="multi_gauss_2d",
                 mu_x_list=None, mu_y_list=None,
                 sigma_list=None,
                 pi_list=None,
                 sample_size=None,
                 xmin=None, xmax=None,
                 ymin=None, ymax=None, 
                 num_bins=None,
                 num_points_x=None,
                 num_points_y=None,
                 target_sample_fname="target.npy"):

        super().__init__(name=name, 
                         sample_size=sample_size,
                         sample_dim=2,
                         xmin=xmin, 
                         xmax=xmax,
                         ymin=ymin,
                         ymax=ymax,
                         target_sample_fname=target_sample_fname)

        self.ymin = ymin
        self.ymax = ymax

        self.mu_x_list = mu_x_list
        self.mu_y_list = mu_y_list
        self.sigma_list = sigma_list

        self.pi_list = pi_list

        param_list = [self.mu_x_list, self.mu_y_list,
                      self.sigma_list,
                      self.pi_list, self.sample_size]

        self.gen_target_prob(num_bins=num_bins)

        if not os.path.isfile(self.target_sample_fname):
            if all(val is not None for val in param_list):
                if (num_bins is not None):
                    self.gen_target_sample()
                elif (num_points_x is not None) and (num_points_y is not None):
                    self.gen_target_sample(num_points_x=num_points_x, num_points_y=num_points_y)
                else:
                    print("Target not generated, missing parameters (num bins or num_points_x/y)")
        else:
            print("Target sample file found, loading from file")

    def gen_target_prob(self, mu_x_list=None, mu_y_list=None,
                          sigma_list=None,
                          pi_list=None,
                          sample_size=None,
                          num_points_x=50,
                          num_points_y=50,
                          num_bins=None,
                          xmin=None, xmax=None,
                          ymin=None, ymax=None):

        mu_x_list = self.mu_x_list if mu_x_list is None else mu_x_list
        self.mu_x_list = mu_x_list

        sigma_list = self.sigma_list if sigma_list is None else sigma_list
        self.sigma_list = sigma_list

        mu_y_list = self.mu_y_list if mu_y_list is None else mu_y_list
        self.mu_y_list = mu_y_list

        pi_list = self.pi_list if pi_list is None else pi_list
        self.pi_list = pi_list

        sample_size = self.sample_size if sample_size is None else sample_size
        self.sample_size = sample_size

        xmin = self.xmin if xmin is None else xmin
        self.xmin = xmin

        xmax = self.xmax if xmax is None else xmax
        self.xmax = xmax

        ymin = self.ymin if ymin is None else ymin
        self.ymin = ymin

        ymax = self.ymax if ymax is None else ymax
        self.ymax = ymax
        
        if num_bins is not None:
            num_points_x = num_bins
            num_points_y = num_bins

        self.num_points_x = num_points_x
        self.num_points_y = num_points_y

        x_arr = np.linspace(self.xmin, self.xmax, num_points_x)
        y_arr = np.linspace(self.ymin, self.ymax, num_points_y)

        x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)

        z_arr = np.zeros((num_points_x, num_points_y))

        for mu_x, mu_y, sigma, pi \
                in zip(mu_x_list, mu_y_list, sigma_list, pi_list):

            for idx, x in np.ndenumerate(x_mesh):
                y = y_mesh[idx]

                z_arr[idx] += pi*np.exp(-1*((x-mu_x)**2 +
                                            (y-mu_y)**2) / 2*sigma**2)

        z_arr /= np.sum(np.sum(z_arr, axis=0), axis=0)

        self.x_arr = x_arr
        self.y_arr = y_arr
        # Need to transpose prob arr to match directionality of ndenumerate and meshgrid
        # Scan across x first
        self.prob_arr = z_arr.T
        
        prob_arr_flat = z_arr.flatten()
        # Reverse tuple to scan across x first
        idx_list = [tuple(reversed(idx)) for idx, _ in np.ndenumerate(x_mesh)]
        idx_arr = np.array(idx_list)
        
        self.target_hist_idx_prob_arr = prob_arr_flat
        self.target_hist_idx_arr = idx_arr
    
    def gen_target_sample_to_file(self):
        self.target_sample_data_h.create_new_file()
            
        with open(self.target_sample_fname, 'wb') as f:
            gen_batch_size = 10000
            
            num_batches = self.sample_size // gen_batch_size
            
            sample_mod_batch = self.sample_size % gen_batch_size
            
            bar = tqdm(range(num_batches))
            
            bar.set_description("Target sample generation")
            
            def gen_samples(num_samples):
                sample_idx_list = np.random.choice(len(self.target_hist_idx_arr), num_samples, 
                                              p=self.target_hist_idx_prob_arr)

                # Only implemented for 2D for now
                x_samples = [self.x_arr[self.target_hist_idx_arr[idx][0]] for idx in sample_idx_list]
                y_samples = [self.y_arr[self.target_hist_idx_arr[idx][1]] for idx in sample_idx_list]        
                
                return torch.from_numpy(np.array(list(zip(x_samples, y_samples))))
            
            for e in bar:
                sample = gen_samples(gen_batch_size)
                 
                self.target_sample_data_h.write_tensor_to_file(tensor=sample)      
                
                del sample
                gc.collect()
            
            if sample_mod_batch > 0: 
                sample = gen_samples(sample_mod_batch)
                
                self.target_sample_data_h.write_tensor_to_file(tensor=sample)            
                                
                del sample
                gc.collect()       
    
    def gen_target_sample(self):
        self.gen_target_sample_to_file()   