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

    mu_x_list = [-2, 2]
    mu_y_list = [0, 0]
    sigma_list = [1, 1]
    pi_list = [1, 1]

    xmin = -2
    xmax = 2
    ymin = -1
    ymax = 1

    num_diffusion_steps = 100
    dt = 0.005
    
    num_hist_bins = 10

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
        forward_diffusion_samples = myDiffNum.forward_diffusion_passive()
        
        forward_first = forward_diffusion_samples[0]
        forward_last = forward_diffusion_samples[-1]
        
        hist_forw_first, xb_forw_first, yb_forw_first = np.histogram2d(forward_first[:,0], forward_first[:,1],
                                                                       density=True,
                                                                       bins=num_hist_bins,
                                                                       range=[[xmin, xmax], [ymin, ymax]])
        
        hist_forw_last, xb_forw_last, yb_forw_last = np.histogram2d(forward_last[:,0], forward_last[:,1],
                                                                density=True,
                                                                bins=num_hist_bins,
                                                                range=[[xmin, xmax], [ymin, ymax]])
        
        myDiffNum.target.gen_target_sample(num_bins=num_hist_bins)
        # myDiffNum.num_diffusion_steps=1
        myDiffNum.train_diffusion_passive(iterations=1000)
        myDiffNum.sample_from_diffusion_passive()
        
        with open(f"{ofile_base}.pkl", 'wb') as f:
            pickle.dump(myDiffNum, f)
              
        rev_first = myDiffNum.passive_reverse_samples[0]
        
        rev_last = myDiffNum.passive_reverse_samples[-1]
        
        hist_rev_first, xb_rev_first, yb_rev_first = np.histogram2d(rev_first[:,0], rev_first[:,1], 
                                                        density=True,
                                                        bins=num_hist_bins,
                                                        range=[[xmin, xmax], [ymin, ymax]])
        
        hist_rev_last, xb_rev_last, yb_rev_last = np.histogram2d(rev_last[:,0], rev_last[:,1], 
                                                     density=True,
                                                     bins=num_hist_bins,
                                                     range=[[xmin, xmax], [ymin, ymax]])
        
        target = myDiffNum.target.sample
        
        hist_target, xb_target, yb_target = np.histogram2d(target[:,0], target[:,1], 
                                                           density=True,
                                                           bins=num_hist_bins,
                                                           range=[[xmin, xmax], [ymin, ymax]])
        
        fig, axs = plt.subplots(3, 2)
        
        fig.set_size_inches(5,7)
        
        axs[0,0].imshow(hist_forw_first) #, extent=[xb_forw_first[0], xb_forw_first[-1],
                                               #  yb_forw_first[0], yb_forw_first[-1]])
        
        axs[0,0].set_title('forw[0]')
        
        axs[0,1].imshow(hist_forw_last) #, extent=[xb_forw_last[0], xb_forw_last[-1],
                                         #     yb_forw_last[0], yb_forw_last[-1]])
        
        axs[0,1].set_title('forw[-1]')
        
        axs[1,0].imshow(hist_rev_first) #, extent=[xb_rev_first[0], xb_rev_first[-1], 
                                        #  yb_rev_first[0], yb_rev_first[-1]])
        
        axs[1,0].set_title('rev[0]')
        
        axs[1,1].imshow(hist_rev_last) #, extent=[xb_rev_last[0], xb_rev_last[-1], 
                                        #  yb_rev_last[0], yb_rev_last[-1]])
        
        axs[1,1].set_title('rev[-1]')
        
        axs[2,0].imshow(hist_target) #, extent=[xb_target[0], xb_target[-1], 
                                      #    yb_target[0], yb_target[-1]])
        
        axs[2,0].set_title('target')
        
        axs[2,1].axis('off')
        
        fig.tight_layout()
        
        plt.savefig("passive_rev.png")
        
        plt.close(fig)
    
    # breakpoint()
