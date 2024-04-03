from diffusion_2D import Diffusion2D

import numpy as np
import torch
from tqdm.auto import tqdm
import itertools
import copy

import multiprocess
from multiprocess import Pool

class DiffusionNumeric2D(Diffusion2D):
    def __init__(self, ofile_base="", passive_noise=None, active_noise=None,
                 target=None, num_diffusion_steps=None, dt=None, k=1,
                 sample_dim=None, data_proc=None):
        
        super().__init__(ofile_base=ofile_base,
                         passive_noise=passive_noise,
                         active_noise=active_noise,
                         target=target,
                         num_diffusion_steps=num_diffusion_steps,
                         dt=dt,
                         k=k,
                         sample_dim=sample_dim,
                         data_proc=data_proc)
        
        self.passive_models = None
        self.passive_loss_history = None
        
    def compute_loss_passive(self, forward_samples, t, score_model):
        
        forward_samples = [torch.from_numpy(f) for f in forward_samples]
        
        sample_t = forward_samples[t].reshape(self.sample_dim, 2).type(torch.DoubleTensor)
        
        l = -(sample_t - forward_samples[0]*np.exp(-t*self.dt)) / \
            (self.passive_noise.temperature * (1 - np.exp(-2*t*self.dt)))
            
        scr = score_model(sample_t)
        
        loss = torch.mean((scr - l)**2)
        
        return loss, torch.mean(l**2), torch.mean(scr**2)
        
    def train_diffusion_passive(self, nrnodes=4, iterations=500):
        loss_history = []
        all_models = []
        
        forward_samples = self.forward_diffusion_passive()
        
        if len(forward_samples[0].shape) == 1:
            num_points = forward_samples[0].shape[0]

            forward_samples = [f.reshape((self.sample_dim, 1))
                               for f in forward_samples]
        else:
            num_points = forward_samples[0].shape[0]
            num_dim = forward_samples[0].shape[1]
            
            forward_samples = [f.reshape((self.sample_dim, num_dim))
                               for f in forward_samples]
        
        bar = tqdm(range(1, self.num_diffusion_steps))
        
        t_idx = 1
        
        time_step_list = []
        
        for e in bar:
            score_model = torch.nn.Sequential(
                torch.nn.Linear(2, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, 2)
            ).double()
            
            optim = torch.optim.AdamW(itertools.chain(
                    score_model.parameters()), lr=1e-2, weight_decay=1e-8,)
            
            for _ in range(iterations):
                optim.zero_grad()

                loss, l, scr = self.compute_loss_passive(forward_samples,
                                                         t_idx,
                                                         score_model)

                loss.backward()
                
                optim.step()

            bar.set_description(f'Time:{t_idx} Loss: {loss.item():.4f}')

            all_models.append(copy.deepcopy(score_model))

            loss_history.append(loss.item())

            time_now = t_idx * self.dt

            time_step_list.append(time_now)

            t_idx += 1

        self.passive_forward_time_arr = np.array(time_step_list)
        self.passive_models = all_models
        self.passive_loss_history = np.array(loss_history)

        return all_models
    
    def sample_from_diffusion_passive(self, all_models=None, time=None):
        if all_models is None:
            all_models = self.passive_models
            
        sample_t = self.passive_noise.temperature * \
            torch.randn(self.sample_dim, 2).type(torch.DoubleTensor)
            
        samples = [sample_t.detach()]
        
        time_step_list = []
        
        if time is None:
            reverse_diffusion_step_start = self.num_diffusion_steps - 2
        else:
            reverse_diffusion_step_start = int(np.ceil(time/self.dt)) - 1

            try:
                if reverse_diffusion_step_start > self.num_diffusion_steps - 2:
                    reverse_diffusion_step_start = self.num_diffusion_steps - 2
                    raise IndexError
            except IndexError:
                print(
                    "Provided time value out of bounds, decreasing to max available time")

        self.num_passive_reverse_diffusion_steps = reverse_diffusion_step_start + 1
        
        for t in range(reverse_diffusion_step_start, 0, -1):
            time_now = t*self.dt

            time_step_list.append(time_now)
    

            F = all_models[t](sample_t)
            # If using the total score function

            sample_t = sample_t + sample_t*self.dt + \
                2*self.passive_noise.temperature*F*self.dt + \
                np.sqrt(2*self.passive_noise.temperature*self.dt) * \
                torch.randn(self.sample_dim, 2)

            samples.append(sample_t.detach())

        self.passive_reverse_time_arr = np.array(time_step_list)

        self.passive_reverse_samples = samples

        return sample_t.detach(), samples