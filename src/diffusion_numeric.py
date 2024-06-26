from diffusion import Diffusion
from data_handler import ModelHandler

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import itertools
import copy

import multiprocess
from multiprocess import Pool
import mmap
import gc
import sys

from torch.distributions.multivariate_normal import MultivariateNormal

class DiffusionNumeric(Diffusion):
    def __init__(self, ofile_base="", 
                 passive_noise=None, 
                 active_noise=None, 
                 target=None,
                 num_diffusion_steps=None, 
                 dt=None, 
                 k=1, 
                 data_proc=None,
                 sample_size=None,
                 nn_batch_size=1000,
                 models_passive_fname="models_passive.pt",
                 models_active_x_fname="models_active_x.pt",
                 models_active_eta_fname="models_active_eta.pt"):

        super().__init__(ofile_base=ofile_base,
                         passive_noise=passive_noise,
                         active_noise=active_noise,
                         target=target,
                         num_diffusion_steps=num_diffusion_steps,
                         dt=dt,
                         k=k,
                         data_proc=data_proc,
                         diffusion_type='numeric',
                         sample_size=sample_size)
        
        if self.sample_size > nn_batch_size:
            self.nn_batch_size = nn_batch_size
        else:
            self.nn_batch_size = self.sample_size
        
        self.models_passive_fname = models_passive_fname
        self.models_passive_data_h = ModelHandler(fname=self.models_passive_fname)
        
        self.loss_history_passive = None

        self.models_active_x_fname = models_active_x_fname
        self.models_active_x_data_h = ModelHandler(fname=self.models_active_x_fname)
        
        self.models_active_eta_fname = models_active_eta_fname
        self.models_active_eta_data_h = ModelHandler(fname=self.models_active_eta_fname)
        
        self.loss_history_active_x = None
        self.loss_history_active_eta = None

    def compute_loss_passive(self, t_idx, forward_samples, forward_samples_zero, score_model):
        sample_t = forward_samples.reshape(self.sample_size, self.sample_dim).type(torch.DoubleTensor)
        sample_0 = forward_samples_zero.reshape(self.sample_size, self.sample_dim).type(torch.DoubleTensor)

        l = -(sample_t - sample_0 * np.exp(-t_idx*self.dt)) / \
             (self.passive_noise.temperature * (1-np.exp(-2*t_idx*self.dt)))

        scr = score_model(sample_t)
        loss = torch.mean((scr - l)**2)

        return loss, torch.mean(l**2), torch.mean(scr**2)

    def train_diffusion_passive(self, nrnodes=4, iterations=500):
        self.models_passive_data_h.create_new_file()
        
        loss_history = []
        
        self.models_passive_data_h.clear_models()
        
        forward_samples = self.forward_passive_data_h.mmap_tensor_from_file()

        bar = tqdm(range(1, self.num_diffusion_steps))

        t_idx = 1

        time_step_list = []

        for e in bar:
            score_model = torch.nn.Sequential(
                torch.nn.Linear(self.sample_dim, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, self.sample_dim)
            ).double()

            optim = torch.optim.AdamW(itertools.chain(
                score_model.parameters()), lr=1e-2, weight_decay=1e-8,)

            for _ in range(iterations):
                
                t = t_idx 
                
                loss, l, scr = self.compute_loss_passive(t_idx, forward_samples[t_idx], forward_samples[0], score_model)
                                
                optim.zero_grad()

                loss.backward()
                optim.step()

            bar.set_description(f'Passive training - Time: {t_idx} Loss: {loss.item():.4f}')

            self.models_passive_data_h.write_model_to_file(model=score_model)

            loss_history.append(loss.item())

            time_now = t_idx * self.dt

            time_step_list.append(time_now)

            t_idx += 1
            
            del score_model
            del optim
            gc.collect()

        self.passive_forward_time_arr = np.array(time_step_list)
        self.loss_history_passive = np.array(loss_history)
        
        del forward_samples
        gc.collect()
  
        self.forward_passive_data_h.close_mmap()
        
        gc.collect()

    def sample_from_diffusion_passive(self, all_models=None, time=None):

        if all_models is None:
            all_models = self.passive_models_data_h.load_models()

        sample_t = self.passive_noise.temperature * \
            torch.normal(torch.zeros(self.sample_size, self.sample_dim),
                         torch.ones(self.sample_size, self.sample_dim)).type(torch.DoubleTensor)

        self.reverse_passive_data_h.create_new_file(fname=self.reverse_passive_data_h.fname)
        self.reverse_passive_data_h.write_tensor_to_file(tensor=sample_t)

        time_step_list = []

        if time is None:
            reverse_diffusion_step_start = self.num_diffusion_steps - 2
        else:
            reverse_diffusion_step_start = int(np.floor(time/self.dt)) - 1

            try:
                if reverse_diffusion_step_start > self.num_diffusion_steps - 2:
                    reverse_diffusion_step_start = self.num_diffusion_steps - 2
                    raise IndexError
            except IndexError:
                print(
                    "Provided time value out of bounds, decreasing to max available time")

        self.num_passive_reverse_diffusion_steps = reverse_diffusion_step_start + 1

        for t_idx in range(reverse_diffusion_step_start, 0, -1):

            time_now = t_idx*self.dt

            time_step_list.append(time_now)

            F = all_models[t_idx](sample_t)
            # If using the total score function

            sample_t = sample_t + sample_t*self.dt + \
                2*self.passive_noise.temperature*F*self.dt + \
                np.sqrt(2*self.passive_noise.temperature*self.dt) * \
                torch.normal(torch.zeros_like(sample_t),
                             torch.ones_like(sample_t))

            self.reverse_passive_data_h.write_tensor_to_file(tensor=sample_t)

        self.passive_reverse_time_arr = np.array(time_step_list)
        
        del all_models
        gc.collect()

    def compute_loss_active(self, t_idx,
                            forward_samples_x, forward_samples_x_zero,
                            forward_samples_eta, forward_samples_eta_zero,
                            score_model_x, score_model_eta):

        a = np.exp(-self.k*t_idx*self.dt)

        b = (np.exp(-t_idx*self.dt/self.active_noise.correlation_time) -
             np.exp(-self.k*t_idx*self.dt))/(self.k-(1/self.active_noise.correlation_time))

        c = np.exp(-t_idx*self.dt/self.active_noise.correlation_time)

        M11, M12, M22 = self.M_11_12_22(t_idx)

        det = M11*M22 - M12*M12
        x0 = forward_samples_x_zero.reshape(self.sample_size, self.sample_dim).type(torch.DoubleTensor)
        eta0 = forward_samples_eta_zero.reshape(self.sample_size, self.sample_dim).type(torch.DoubleTensor)
        x = forward_samples_x.reshape(self.sample_size, self.sample_dim).type(torch.DoubleTensor) #[t_idx].type(torch.DoubleTensor)
        eta = forward_samples_eta.reshape(self.sample_size, self.sample_dim).type(torch.DoubleTensor) #[t_idx].type(torch.DoubleTensor)

        Fx = (1/det)*(-M22*(x - a*x0 - b*eta0) + M12*(eta - c*eta0))
        Feta = (1/det)*(-M11*(eta - c*eta0) + M12*(x - a*x0 - b*eta0))

        if len(Fx.shape) == 1:
            Fx = Fx.reshape((Fx.shape[0], 1))
        if len(Feta.shape) == 1:
            Feta = Feta.reshape((Feta.shape[0], 1))
        if len(x.shape) == 1:
            x = x.reshape((x.shape[0], 1))
        if len(eta.shape) == 1:
            eta = eta.reshape((eta.shape[0], 1))

        F = torch.cat((Fx, Feta), dim=1)
        xin = torch.cat((x, eta), dim=1)
        scr_x = score_model_x(xin)
        scr_eta = score_model_eta(xin)
        loss_x = torch.mean((scr_x - Fx)**2)
        loss_eta = torch.mean((scr_eta - Feta)**2)

        return loss_x, loss_eta, torch.mean(Fx**2), torch.mean(Feta**2), torch.mean(scr_x**2), torch.mean(scr_eta**2)

    def M_11_12_22(self, t_idx):
        t = t_idx*self.dt

        a = np.exp(-self.k*t)
        b = np.exp(-t/self.active_noise.correlation_time)

        Tx = self.active_noise.temperature.passive
        Ty = self.active_noise.temperature.active / \
            (self.active_noise.correlation_time**2)
        w = (1/self.active_noise.correlation_time)

        M11 = (1/self.k)*Tx*(1-a**2) + (1/self.k)*Ty*(1/(w*(self.k+w)) + 4*a*b *
                                                      self.k/((self.k+w)*(self.k-w)**2) - (self.k*b**2 + w*a**2)/(w*(self.k-w)**2))
        M12 = (Ty/(w*(self.k**2 - w**2))) * \
            (self.k*(1-b**2) - w*(1 + b**2 - 2*a*b))
        M22 = (Ty/w)*(1-b**2)

        return M11, M12, M22

    def train_diffusion_active(self, nrnodes=4, iterations=500):
        self.models_active_x_data_h.create_new_file()
        self.models_active_eta_data_h.create_new_file()
        
        loss_history_x = []
        loss_history_eta = []
        
        forward_samples_x = self.forward_active_x_data_h.mmap_tensor_from_file()
        forward_samples_eta = self.forward_active_eta_data_h.mmap_tensor_from_file()

        t_idx = 1

        time_step_list = []

        bar = tqdm(range(1, self.num_diffusion_steps))

        for e in bar:
            score_model_x = torch.nn.Sequential(
                torch.nn.Linear(self.sample_dim*2, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, self.sample_dim)
            ).double()

            score_model_eta = torch.nn.Sequential(
                torch.nn.Linear(self.sample_dim*2, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, self.sample_dim)
            ).double()

            optim_x = torch.optim.AdamW(itertools.chain(
                score_model_x.parameters()), lr=1e-2, weight_decay=1e-8,)

            optim_eta = torch.optim.AdamW(itertools.chain(
                score_model_eta.parameters()), lr=1e-2, weight_decay=1e-8,)

            for _ in range(iterations):
                    
                t = t_idx 
                
                optim_x.zero_grad()
                optim_eta.zero_grad()

                loss_x, loss_eta, loss_Fx, loss_Feta, loss_scr_x, loss_scr_eta = \
                    self.compute_loss_active(t_idx, 
                                                forward_samples_x[t_idx],
                                                forward_samples_x[0],
                                                forward_samples_eta[t_idx],
                                                forward_samples_eta[0],
                                            score_model_x,
                                            score_model_eta)

                loss_x.backward()
                loss_eta.backward()

                optim_x.step()
                optim_eta.step()

            bar.set_description(
                f'Active training - Time: {t_idx} Loss: {loss_x.item():.4f} Fx: {loss_Fx.item():.4f} scr_x: {loss_scr_x.item():.4f}')

            self.models_active_x_data_h.write_model_to_file(model=score_model_x)
            self.models_active_eta_data_h.write_model_to_file(model=score_model_eta)

            loss_history_x.append(loss_x.item())
            loss_history_eta.append(loss_eta.item())

            time_now = t_idx * self.dt

            time_step_list.append(time_now)

            t_idx += 1
            
            del score_model_x
            del score_model_eta
            del optim_x
            del optim_eta
            gc.collect()

        self.active_forward_time_arr = np.array(time_step_list)

        self.loss_history_active_x = np.array(loss_history_x)
        self.loss_history_active_eta = np.array(loss_history_eta)
        
        del forward_samples_x
        del forward_samples_eta
        gc.collect()
        
        self.forward_active_x_data_h.close_mmap()
        self.forward_active_eta_data_h.close_mmap()
        
        gc.collect()

    def sample_from_diffusion_active(self, all_models_x=None, all_models_eta=None, time=None):        
        if all_models_x is None:
            all_models_x = self.active_models_x_data_h.load_models()

        if all_models_eta is None:
            all_models_eta = self.active_models_eta_data_h.load_models()

        Tp = self.active_noise.temperature.passive
        Ta = self.active_noise.temperature.active
        tau = self.active_noise.correlation_time
        
        var_11 = 1/self.k * (Tp + Ta/(1+ self.k*tau))
        var_12 = Ta/(1 + tau*self.k)
        var_22 = Ta / tau
        
        zero_mean = torch.zeros(2)
        
        covar = torch.tensor([var_11, var_12, var_12, var_22], device=zero_mean.device)
        covar = torch.reshape(covar, (2,2))
        
        sampler = MultivariateNormal(loc=zero_mean, covariance_matrix=covar)
        
        sample = sampler.sample(sample_shape=torch.Size([self.sample_size]))
        
        sample_x, sample_eta = torch.chunk(sample, 2, dim=1)
        
        self.reverse_active_x_data_h.create_new_file(fname=self.reverse_active_x_data_h.fname)
        self.reverse_active_x_data_h.write_tensor_to_file(tensor=sample_x)

        self.reverse_active_eta_data_h.create_new_file(fname=self.reverse_active_eta_data_h.fname)
        self.reverse_active_eta_data_h.write_tensor_to_file(tensor=sample_eta)
        
        x = sample_x.type(torch.DoubleTensor)
        eta = sample_eta.type(torch.DoubleTensor)

        self.reverse_active_x_data_h.create_new_file(fname=self.reverse_active_x_data_h.fname)
        self.reverse_active_x_data_h.write_tensor_to_file(tensor=x)

        self.reverse_active_eta_data_h.create_new_file(fname=self.reverse_active_eta_data_h.fname)
        self.reverse_active_eta_data_h.write_tensor_to_file(tensor=eta)
            
        time_step_list = []

        if time is None:
            reverse_diffusion_step_start = self.num_diffusion_steps - 2
        else:
            reverse_diffusion_step_start = int(np.floor(time/self.dt)) - 1

        self.num_active_rverse_diffusion_steps = reverse_diffusion_step_start + 1

        for t_idx in range(reverse_diffusion_step_start, 0, -1):

            time_now = t_idx*self.dt
            time_step_list.append(time_now)
            xin = torch.cat((x, eta), dim=1)

            Fx = all_models_x[t_idx](xin)
            Feta = all_models_eta[t_idx](xin)

            x = x + self.dt*(x-eta) + \
                2*self.active_noise.temperature.passive*Fx*self.dt + \
                np.sqrt(2*self.active_noise.temperature.passive*self.dt) * \
                torch.normal(torch.zeros_like(x),
                             torch.ones_like(x))

            eta = eta + self.dt*eta/self.active_noise.correlation_time + \
                (2*self.active_noise.temperature.active /
                    (self.active_noise.correlation_time**2)) * Feta*self.dt + \
                (1/self.active_noise.correlation_time) * \
                np.sqrt(2 * self.active_noise.temperature.active * self.dt) * \
                torch.normal(torch.zeros_like(eta),
                             torch.ones_like(eta))

            self.reverse_active_x_data_h.write_tensor_to_file(tensor=x)
            self.reverse_active_eta_data_h.write_tensor_to_file(tensor=eta)

        self.active_reverse_time_arr = np.array(time_step_list)
        
        gc.collect()
