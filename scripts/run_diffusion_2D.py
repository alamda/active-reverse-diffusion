import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys

sys.path.insert(0, '../src/')

from data_proc_2D import DataProc2D
from target_multi_gaussian_2D import TargetMultiGaussian2D
from noise import NoiseActive, NoisePassive
from diffusion_numeric import DiffusionNumeric
from diffusion_numeric_2D import DiffusionNumeric2D



if __name__ == "__main__":
    ofile_base = "data"

    sample_dim = 10000

    passive_noise_T = 1.0

    active_noise_Tp = 0.0
    active_noise_Ta = 1.0
    active_noise_tau = 0.25

    mu_x_list = [-1.2, 1.2]
    mu_y_list = [0, 0]
    sigma_list = [1, 1]
    pi_list = [1, 1]

    xmin = -2
    xmax = 2
    ymin = -1
    ymax = 1

    num_diffusion_steps = 50
    dt = 0.005
    
    num_hist_bins = 50

    myPassiveNoise = NoisePassive(T=passive_noise_T,
                                  dim=sample_dim)

    myActiveNoise = NoiseActive(Tp=active_noise_Tp,
                                Ta=active_noise_Ta,
                                tau=active_noise_tau,
                                dim=sample_dim)

    myTarget = TargetMultiGaussian2D(mu_x_list=mu_x_list,
                                     mu_y_list=mu_y_list,
                                     sigma_list=sigma_list,
                                     pi_list=pi_list,
                                     dim=sample_dim,
                                     xmin=xmin, xmax=xmax,
                                     ymin=ymin, ymax=ymax)

    myTarget.gen_target_sample()

    myDataProc = DataProc2D(xmin=xmin, xmax=xmax, 
                            ymin=ymin, ymax=ymax, 
                            num_hist_bins=num_hist_bins)

    myDiffNum = DiffusionNumeric2D(ofile_base=ofile_base,
                                 passive_noise=myPassiveNoise,
                                 active_noise=myActiveNoise,
                                 target=myTarget,
                                 num_diffusion_steps=num_diffusion_steps,
                                 dt=dt,
                                 sample_dim=sample_dim,
                                 data_proc=myDataProc)
    
    if False:
        ps = myDiffNum.forward_diffusion_passive()
        
        fig, axs = plt.subplots(1,2)
        
        ps_first = ps[0]
        ps_last = ps[-1]
        
        hist_first, xb_first, yb_first = np.histogram2d(ps_first[:,0], ps_first[:,1], 
                                                        density=True,
                                                        bins=num_hist_bins,
                                                        range=[[xmin, xmax], [ymin, ymax]])
        
        hist_last, xb_last, yb_last = np.histogram2d(ps_last[:,0], ps_last[:,1], 
                                                     density=True,
                                                     bins=num_hist_bins,
                                                     range=[[xmin, xmax], [ymin, ymax]])
        
        axs[0].imshow(hist_first, extent=[xb_first[0], xb_first[-1], 
                                          yb_first[0], yb_first[-1]])
        
        axs[1].imshow(hist_last, extent=[xb_last[0], xb_last[-1], 
                                          yb_last[0], yb_last[-1]])
        
        for ax in axs:
            ax.set_aspect('equal')
        
        plt.show()
        
        plt.close(fig)
        
    if True:
        myDiffNum.train_diffusion_passive()
        myDiffNum.sample_from_diffusion_passive()
        
        fig, axs = plt.subplots(3,1)
        
        fig.set_size_inches(5,10)
        
        rev_first = np.column_stack((myDiffNum.passive_reverse_samples_x[0],
                                    myDiffNum.passive_reverse_samples_y[0]))
        
        rev_last = np.column_stack((myDiffNum.passive_reverse_samples_x[-1],
                                   myDiffNum.passive_reverse_samples_y[-1]))
        
        hist_first, xb_first, yb_first = np.histogram2d(rev_first[:,0], rev_first[:,1], 
                                                        density=True,
                                                        bins=num_hist_bins,
                                                        range=[[xmin, xmax], [ymin, ymax]])
        
        hist_last, xb_last, yb_last = np.histogram2d(rev_last[:,0], rev_last[:,1], 
                                                     density=True,
                                                     bins=num_hist_bins,
                                                     range=[[xmin, xmax], [ymin, ymax]])
        
        target = myDiffNum.target.sample
        
        hist_target, xb_target, yb_target = np.histogram2d(target[:,0], target[:,1], 
                                                           density=True,
                                                           bins=num_hist_bins,
                                                           range=[[xmin, xmax], [ymin, ymax]])
        
        axs[0].imshow(hist_first, extent=[xb_first[0], xb_first[-1], 
                                          yb_first[0], yb_first[-1]])
        
        axs[0].set_title('rev[0]')
        
        axs[1].imshow(hist_last, extent=[xb_last[0], xb_last[-1], 
                                          yb_last[0], yb_last[-1]])
        
        axs[1].set_title('rev[-1]')
        
        axs[2].imshow(hist_target, extent=[xb_target[0], xb_target[-1], 
                                          yb_target[0], yb_target[-1]])
        
        axs[2].set_title('target')
        
        for ax in axs:
            ax.set_aspect('equal')
            
        fig.tight_layout()
        
        plt.savefig("passive_rev.png")
        
        plt.close(fig)
    
    # breakpoint()
