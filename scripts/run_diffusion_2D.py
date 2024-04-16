import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys

sys.path.insert(0, '../src/')

from data_proc import DataProc
from target_multi_gaussian_2D import TargetMultiGaussian2D
from noise import NoiseActive, NoisePassive
from diffusion_numeric import DiffusionNumeric

if __name__ == "__main__":
    ofile_base = "data"

    sample_size = 100000000

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

    num_diffusion_steps = 10
    dt = 0.005
    
    num_hist_bins = 10
    
    num_passive_iterations=100
    num_active_iterations=100

    myPassiveNoise = NoisePassive(T=passive_noise_T,
                                  dim=sample_size)

    myActiveNoise = NoiseActive(Tp=active_noise_Tp,
                                Ta=active_noise_Ta,
                                tau=active_noise_tau,
                                dim=sample_size)

    myTarget = TargetMultiGaussian2D(mu_x_list=mu_x_list,
                                     mu_y_list=mu_y_list,
                                     sigma_list=sigma_list,
                                     pi_list=pi_list,
                                     sample_size=sample_size,
                                     xmin=xmin, xmax=xmax,
                                     ymin=ymin, ymax=ymax,
                                     num_bins=num_hist_bins)

    myDataProc = DataProc(xmin=xmin, xmax=xmax, 
                          ymin=ymin, ymax=ymax, 
                          num_hist_bins=num_hist_bins)

    myDiffNum = DiffusionNumeric(ofile_base=ofile_base,
                                 passive_noise=myPassiveNoise,
                                 active_noise=myActiveNoise,
                                 target=myTarget,
                                 num_diffusion_steps=num_diffusion_steps,
                                 dt=dt,
                                 sample_size=sample_size,
                                 data_proc=myDataProc)

     
    if True:
        forward_diffusion_samples_pass = myDiffNum.forward_diffusion_passive()
        forward_diffusion_samples_act = myDiffNum.forward_diffusion_active()
        
        forward_pass_first = forward_diffusion_samples_pass[0]
        forward_pass_last = forward_diffusion_samples_pass[-1]
        
        hist_forw_pass_first, xb_forw_pass_first, yb_forw_pass_first = np.histogram2d(forward_pass_first[:,0], 
                                                                                      forward_pass_first[:,1],
                                                                       density=True,
                                                                       bins=num_hist_bins,
                                                                       range=[[xmin, xmax], [ymin, ymax]])
        
        hist_forw_pass_last, xb_forw_pass_last, yb_forw_pass_last = np.histogram2d(forward_pass_last[:,0], 
                                                                                   forward_pass_last[:,1],
                                                                density=True,
                                                                bins=num_hist_bins,
                                                                range=[[xmin, xmax], [ymin, ymax]])
        
        forward_act, _ = forward_diffusion_samples_act
        
        forward_act_first = forward_act[0]
        forward_act_last = forward_act[-1]

        hist_forw_act_first, xb_forw_act_first, yb_forw_act_first = np.histogram2d(forward_act_first[:,0], 
                                                                                      forward_act_first[:,1],
                                                                       density=True,
                                                                       bins=num_hist_bins,
                                                                       range=[[xmin, xmax], [ymin, ymax]])
        
        hist_forw_act_last, xb_forw_act_last, yb_forw_act_last = np.histogram2d(forward_act_last[:,0], 
                                                                                   forward_act_last[:,1],
                                                                density=True,
                                                                bins=num_hist_bins,
                                                                range=[[xmin, xmax], [ymin, ymax]])
        
        
        # myDiffNum.num_diffusion_steps=1

        if os.path.isfile(f"{ofile_base}.pkl"):
            with open(f"{ofile_base}.pkl", 'rb') as f:
                myDiffNum = pickle.load(f)
        else:
            myDiffNum.train_diffusion_passive(iterations=num_passive_iterations)
            myDiffNum.sample_from_diffusion_passive()
            myDiffNum.calculate_passive_diff_list()
            
            myDiffNum.train_diffusion_active(iterations=num_active_iterations)
            myDiffNum.sample_from_diffusion_active()
            myDiffNum.calculate_active_diff_list()
            
            with open(f"{ofile_base}.pkl", 'wb') as f:
                pickle.dump(myDiffNum, f)
              
        rev_pass_first = myDiffNum.passive_reverse_samples[0]
        
        rev_pass_last = myDiffNum.passive_reverse_samples[-1]
        
        hist_rev_pass_first, xb_rev_pass_first, yb_rev_pass_first = np.histogram2d(rev_pass_first[:,0], 
                                                                                   rev_pass_first[:,1], 
                                                        density=True,
                                                        bins=num_hist_bins,
                                                        range=[[xmin, xmax], [ymin, ymax]])
        
        hist_rev_pass_last, xb_rev_pass_last, yb_rev_pass_last = np.histogram2d(rev_pass_last[:,0], 
                                                                                rev_pass_last[:,1], 
                                                     density=True,
                                                     bins=num_hist_bins,
                                                     range=[[xmin, xmax], [ymin, ymax]])
        
        
        rev_act_first = myDiffNum.active_reverse_samples_x[0]
        rev_act_last = myDiffNum.active_reverse_samples_x[-1]
        
        hist_rev_act_first, xb_rev_act_first, yb_rev_act_first = np.histogram2d(rev_act_first[:,0], 
                                                                                   rev_act_first[:,1], 
                                                        density=True,
                                                        bins=num_hist_bins,
                                                        range=[[xmin, xmax], [ymin, ymax]])
        
        hist_rev_act_last, xb_rev_act_last, yb_rev_act_last = np.histogram2d(rev_act_last[:,0], 
                                                                                rev_act_last[:,1], 
                                                     density=True,
                                                     bins=num_hist_bins,
                                                     range=[[xmin, xmax], [ymin, ymax]])
        
        target = myDiffNum.target.sample
        
        hist_target, xb_target, yb_target = np.histogram2d(target[:,0], target[:,1], 
                                                           density=True,
                                                           bins=num_hist_bins,
                                                           range=[[xmin, xmax], [ymin, ymax]])
        
        ###
        
        fig, axs = plt.subplots(3, 5)
        
        fig.set_size_inches(10,10)
        
        axs[0,0].imshow(hist_forw_pass_first) 
        axs[0,0].set_title('pass_forw[0]')
        
        axs[0,1].imshow(hist_forw_pass_last)
        axs[0,1].set_title('pass_forw[-1]')
        
        axs[0,2].axis('off')
        
        axs[0,3].imshow(hist_forw_act_first)
        axs[0,3].set_title('act_forw[0]')
        
        axs[0,4].imshow(hist_forw_act_last)
        axs[0,4].set_title('act_forw[-1]')
        
        ###
        
        axs[1,0].imshow(hist_rev_pass_first)
        axs[1,0].set_title('pass_rev[0]')
        
        axs[1,1].imshow(hist_rev_pass_last)
        axs[1,1].set_title('pass_rev[-1]')
        
        axs[1,2].axis('off')
        
        axs[1,3].imshow(hist_rev_act_first)
        axs[1,3].set_title('act_rev[0]')
                
        axs[1,4].imshow(hist_rev_act_last)
        axs[1,4].set_title('act_rev[-1]')
        
        ###
        
        axs[2,0].imshow(hist_target)
        axs[2,0].set_title('target')
        
        axs[2,1].axis('off')
        axs[2,2].axis('off')
        axs[2,3].axis('off')
        axs[2,4].axis('off')
        
        fig.tight_layout()
        
        plt.savefig("passive_rev.png")
        
        plt.close(fig)

    if True:
        with open("data.pkl", 'rb') as f:
            myDiffNum = pickle.load(f)
        
        fig, ax = plt.subplots()
        
        t_list_passive = np.arange(
            0, len(myDiffNum.passive_diff_list))*myDiffNum.dt
        
        t_list_active = np.arange(0, len(myDiffNum.active_diff_list))*myDiffNum.dt
        
        ax.scatter(t_list_passive, 
                   np.log(myDiffNum.passive_diff_list),
                   label="passive")
        
        ax.scatter(t_list_active, 
                   np.log(myDiffNum.active_diff_list),
                   label="active")
        
        ax.legend()
        
        plt.savefig('diff.png')
        
        plt.close(fig)
    
