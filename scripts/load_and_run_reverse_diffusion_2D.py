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
from memory_profiler import profile

import matplotlib
matplotlib.use('Agg')

from matplotlib import figure

def load_file(fname):
    with open(fname, "rb") as f:
        var = pickle.load(f)
    
    return var

def plot_hist(sample_tensor):
    fig = figure.Figure()
    ax = fig.subplots(1)
    
    hist, _, _ = np.histogram2d(sample_tensor[:,0], sample_tensor[:,1], density=True)
    
    im = ax.imshow(hist)
    
    fig.savefig("hist.png")

def main():
    
    myDiffNum = load_file("data.pkl")
    
    target_sample = myDiffNum.target.target_sample_data_h.mmap_tensor_from_file(fname="target.npy")
    
    forward_passive = myDiffNum.forward_passive_data_h.mmap_tensor_from_file("forward_passive.npy")
    
    forward_x_active = myDiffNum.forward_x_active_data_h.mmap_tensor_from_file("forward_x_active.npy")
    forward_eta_active = myDiffNum.forward_eta_active_data_h.mmap_tensor_from_file("forward_eta_active.npy")
    
    reverse_passive = myDiffNum.reverse_passive_data_h.mmap_tensor_from_file("reverse_passive.npy")
    
    reverse_x_active = myDiffNum.reverse_x_active_data_h.mmap_tensor_from_file("reverse_x_active.npy")
    reverse_eta_active = myDiffNum.reverse_eta_active_data_h.mmap_tensor_from_file("reverse_eta_active.npy")


    passive_models = load_file("passive_models.pt")

    active_models_x = load_file("active_models_x.pt")
    active_models_eta = load_file("active_models_eta.pt")
    
    myDiffNum.target.sample = target_sample
    myDiffNum.passive_models = passive_models
    myDiffNum.active_models_x = active_models_x
    myDiffNum.active_models_eta = active_models_eta
    

    time_list = [0.010, 0.020, 0.030, 0.040]
    
    
    pass_rev_time_list = []
    pass_rev_diff_list = []
    act_rev_time_list = []
    act_rev_diff_list = []
    
    for time in time_list:
        myDiffNum.sample_from_diffusion_passive(time=time)
        myDiffNum.calculate_passive_diff_list(multiproc=False)
        
        pass_rev_time_list.append(myDiffNum.passive_reverse_time_arr[-1])
        pass_rev_diff_list.append(myDiffNum.passive_diff_list[-1])
        
        myDiffNum.sample_from_diffusion_active(time=time)
        myDiffNum.calculate_active_diff_list(multiproc=False)
        
        act_rev_time_list.append(myDiffNum.active_reverse_time_arr[-1])
        act_rev_diff_list.append(myDiffNum.active_diff_list[-1])
        
    print(pass_rev_time_list)
    print(pass_rev_diff_list)
    
    print(act_rev_time_list)
    print(act_rev_diff_list)
    
    fig = figure.Figure()
    ax = fig.subplots(1)
    
    ax.scatter(np.array(pass_rev_time_list), np.log(np.array(pass_rev_diff_list)), label="passive")
    ax.scatter(np.array(act_rev_time_list), np.log(np.array(act_rev_diff_list)), label="active")
    
    ax.legend()
    
    fig.savefig("crossover.png")

if __name__=="__main__":
    main()
