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
        
    def compute_loss_passive(self, forward_samples, t, score_model_x, score_model_y):
        
        sample_x_t = forward_samples[t][:,0].reshape(self.sample_dim, 1).type(torch.DoubleTensor)
        
        sample_y_t = forward_samples[t][:,1].reshape(self.sample_dim, 1).type(torch.DoubleTensor)
        
        l_x = -(sample_x_t - forward_samples[0][:,0]*np.exp(-t*self.dt)) / \
            (self.passive_noise.temperature * (1-np.exp(-2*t*self.dt)))
            
        l_y = -(sample_y_t - forward_samples[0][:,1]*np.exp(-t*self.dt)) / \
            (self.passive_noise.temperature * (1-np.exp(-2*t*self.dt)))

        sample_t = torch.cat((sample_x_t, sample_y_t), dim=1)

        scr_x = score_model_x(sample_t)
        scr_y = score_model_y(sample_t)
        
        loss_x = torch.mean((scr_x - l_x)**2)
        loss_y = torch.mean((scr_y - l_y)**2)
        
        return loss_x, loss_y, torch.mean(l_x**2), torch.mean(scr_x**2)
    
    def train_diffusion_passive(self, nrnodes=4, iterations=500):
        loss_history = []
        all_models_x = []
        all_models_y = []
        
        forward_samples = self.forward_diffusion_passive()
        
        if len(forward_samples[0].shape) == 1:
            num_points = forward_samples[0].shape[0]

            forward_samples = [torch.from_numpy(f).reshape((self.sample_dim, 1))
                               for f in forward_samples]
        else:
            num_points = forward_samples[0].shape[0]
            num_dim = forward_samples[0].shape[1]
            
            forward_samples = [torch.from_numpy(f).reshape((self.sample_dim, num_dim))
                               for f in forward_samples]
        
        bar = tqdm(range(1, self.num_diffusion_steps))
        
        t_idx = 1
        
        time_step_list = []
        
        for e in bar:
            score_model_x = torch.nn.Sequential(
                torch.nn.Linear(2, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, 1)
            ).double()
            
            score_model_y = torch.nn.Sequential(
                torch.nn.Linear(2, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, 1)
            ).double()
            
            optim_x = torch.optim.AdamW(itertools.chain(
                    score_model_x.parameters()), lr=1e-2, weight_decay=1e-8,)

            optim_y = torch.optim.AdamW(itertools.chain(
                    score_model_y.parameters()), lr=1e-2, weight_decay=1e-8,)

            loss = 100

            for _ in range(iterations):
                optim_x.zero_grad()
                optim_y.zero_grad()

                loss_x, loss_y, l, scr = self.compute_loss_passive(forward_samples,
                                                         t_idx,
                                                         score_model_x, 
                                                         score_model_y)

                loss_x.backward()
                loss_y.backward()
                
                optim_x.step()
                optim_y.step()

            bar.set_description(f'Time:{t_idx} Loss: {loss_x.item():.4f}')

            all_models_x.append(copy.deepcopy(score_model_x))
            all_models_y.append(copy.deepcopy(score_model_y))

            loss_history.append(loss_x.item())

            time_now = t_idx * self.dt

            time_step_list.append(time_now)

            t_idx += 1

        self.passive_forward_time_arr = np.array(time_step_list)
        self.passive_models_x = all_models_x
        self.passive_models_y = all_models_y
        self.passive_loss_history = np.array(loss_history)

        return all_models_x
    
    def sample_from_diffusion_passive(self, all_models_x=None, all_models_y=None, time=None):
        if all_models_x is None:
            all_models_x = self.passive_models_x
            
        if all_models_y is None:
            all_models_y = self.passive_models_y
            
        sample_x_t = self.passive_noise.temperature * \
            torch.randn(self.sample_dim, 1).type(torch.DoubleTensor)
            
        sample_y_t = self.passive_noise.temperature * \
            torch.randn(self.sample_dim, 1).type(torch.DoubleTensor)
            
        samples_x = [sample_x_t.detach()]
        samples_y = [sample_y_t.detach()]
        
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
            
            sample_t = torch.cat((sample_x_t, sample_y_t), dim=1)

            Fx = all_models_x[t](sample_t)
            Fy = all_models_y[t](sample_t)
            # If using the total score function

            sample_x_t = sample_x_t + sample_x_t*self.dt + \
                2*self.passive_noise.temperature*Fx*self.dt + \
                np.sqrt(2*self.passive_noise.temperature*self.dt) * \
                torch.randn(self.sample_dim, 1)
                
            sample_y_t = sample_y_t + sample_y_t*self.dt + \
                2*self.passive_noise.temperature*Fy*self.dt + \
                np.sqrt(2*self.passive_noise.temperature*self.dt) * \
                torch.randn(self.sample_dim, 1)

            samples_x.append(sample_x_t.detach())
            samples_y.append(sample_y_t.detach())

        self.passive_reverse_time_arr = np.array(time_step_list)

        self.passive_reverse_samples_x = samples_x
        self.passive_reverse_samples_y = samples_y

        return sample_x_t.detach(), samples_x