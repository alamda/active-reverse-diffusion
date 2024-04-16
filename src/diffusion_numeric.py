from diffusion import Diffusion

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import itertools
import copy

import multiprocess
from multiprocess import Pool

import mmap

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
                 nn_batch_size=1000):

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
        
        self.passive_models = None
        self.passive_loss_history = None

        self.active_models_x = None
        self.active_models_eta = None
        self.active_loss_history_x = None
        self.active_loss_history_eta = None

    def compute_loss_passive(self, t_idx, forward_samples, forward_samples_zero, score_model):
        sample_t = forward_samples.reshape(self.nn_batch_size, self.sample_dim).type(torch.DoubleTensor)
        sample_0 = forward_samples_zero.reshape(self.nn_batch_size, self.sample_dim).type(torch.DoubleTensor)

        l = -(sample_t - sample_0 * np.exp(-t_idx*self.dt)) / \
             (self.passive_noise.temperature * (1-np.exp(-2*t_idx*self.dt)))

        scr = score_model(sample_t)
        loss = torch.mean((scr - l)**2)

        return loss, torch.mean(l**2), torch.mean(scr**2)

    def train_diffusion_passive(self, nrnodes=4, iterations=500):
        loss_history = []
        all_models = []

        forward_samples = self.forward_diffusion_passive()

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
            
            dataset = TensorDataset(forward_samples[t_idx], forward_samples[0])
            dataloader = DataLoader(dataset, batch_size=self.nn_batch_size, shuffle=True)

            for _ in range(iterations):
                for id_batch, (sample_batch, sample_zero_batch) in enumerate(dataloader):

                    sample_batch = sample_batch.reshape(self.nn_batch_size, self.sample_dim).type(torch.DoubleTensor)
                    sample_zero_batch = sample_zero_batch.reshape(self.nn_batch_size, self.sample_dim).type(torch.DoubleTensor)
                    
                    t = t_idx 
                    
                    loss, l, scr = self.compute_loss_passive(t_idx, sample_batch, sample_zero_batch, score_model)
                                 
                    optim.zero_grad()

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
            torch.randn(self.sample_size, self.sample_dim).type(torch.DoubleTensor)

        with open(self.reverse_passive_fname, "wb") as f:
            f.write(sample_t.detach().numpy().tobytes())

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

        for t_idx in range(reverse_diffusion_step_start, 0, -1):

            time_now = t_idx*self.dt

            time_step_list.append(time_now)

            F = all_models[t_idx](sample_t)
            # If using the total score function

            sample_t = sample_t + sample_t*self.dt + \
                2*self.passive_noise.temperature*F*self.dt + \
                np.sqrt(2*self.passive_noise.temperature*self.dt) * \
                torch.randn(self.sample_size, self.sample_dim)

            with open(self.reverse_passive_fname, "ab") as f:
                f.write(sample_t.detach().numpy().tobytes())

        self.passive_reverse_time_arr = np.array(time_step_list)

        with open(self.reverse_passive_fname, "r+b") as f:
            mm_r = mmap.mmap(f.fileno(), 0)
            mm_r_arr = np.frombuffer(mm_r, dtype=np.double)
            
            _ = mm_r_arr.shape
            
            self.passive_reverse_samples = \
                torch.from_numpy(np.reshape(mm_r_arr, \
                    (len(time_step_list)+1, self.sample_size, self.sample_dim)))
                
            self.passive_reverse_samples = [x for x in self.passive_reverse_samples]

        return sample_t.detach(), self.passive_reverse_samples

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
        x0 = forward_samples_x_zero.reshape(self.nn_batch_size, self.sample_dim).type(torch.DoubleTensor)
        eta0 = forward_samples_eta_zero.reshape(self.nn_batch_size, self.sample_dim).type(torch.DoubleTensor)
        x = forward_samples_x.reshape(self.nn_batch_size, self.sample_dim).type(torch.DoubleTensor) #[t_idx].type(torch.DoubleTensor)
        eta = forward_samples_eta.reshape(self.nn_batch_size, self.sample_dim).type(torch.DoubleTensor) #[t_idx].type(torch.DoubleTensor)

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
        loss_history_x = []
        loss_history_eta = []

        all_models_x = []
        all_models_eta = []

        forward_samples_x, forward_samples_eta = self.forward_diffusion_active()

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

            dataset = TensorDataset(forward_samples_x[t_idx], forward_samples_x[0],
                                    forward_samples_eta[t_idx], forward_samples_eta[0])
            dataloader = DataLoader(dataset, batch_size=self.nn_batch_size, shuffle=True)

            for _ in range(iterations):
                for id_batch, (sample_x_batch, sample_x_zero_batch, \
                               sample_eta_batch, sample_eta_zero_batch) \
                                   in enumerate(dataloader):
                    
                    sample_x_batch = sample_x_batch.reshape(self.nn_batch_size, self.sample_dim).type(torch.DoubleTensor)
                    sample_x_zero_batch = sample_x_zero_batch.reshape(self.nn_batch_size, self.sample_dim).type(torch.DoubleTensor)
                    
                    sample_eta_batch = sample_eta_batch.reshape(self.nn_batch_size, self.sample_dim).type(torch.DoubleTensor)
                    sample_eta_zero_batch = sample_eta_zero_batch.reshape(self.nn_batch_size, self.sample_dim).type(torch.DoubleTensor)
                    
                    t = t_idx 
                    
                    optim_x.zero_grad()
                    optim_eta.zero_grad()

                    loss_x, loss_eta, loss_Fx, loss_Feta, loss_scr_x, loss_scr_eta = \
                        self.compute_loss_active(t_idx, 
                                                 sample_x_batch,
                                                 sample_x_zero_batch,
                                                sample_eta_batch,
                                                sample_eta_zero_batch,
                                                score_model_x,
                                                score_model_eta)

                    loss_x.backward()
                    loss_eta.backward()

                    optim_x.step()
                    optim_eta.step()

            bar.set_description(
                f'Time:{t_idx} Loss: {loss_x.item():.4f} Fx: {loss_Fx.item():.4f} scr_x: {loss_scr_x.item():.4f}')

            all_models_x.append(copy.deepcopy(score_model_x))
            all_models_eta.append(copy.deepcopy(score_model_eta))

            loss_history_x.append(loss_x.item())
            loss_history_eta.append(loss_eta.item())

            time_now = t_idx * self.dt

            time_step_list.append(time_now)

            t_idx += 1

        self.active_forward_time_arr = np.array(time_step_list)

        self.active_models_x = all_models_x
        self.active_models_eta = all_models_eta

        self.active_loss_history_x = np.array(loss_history_x)
        self.active_loss_history_eta = np.array(loss_history_eta)

        return all_models_x, all_models_eta

    def sample_from_diffusion_active(self, all_models_x=None, all_models_eta=None, time=None):
        if all_models_x is None:
            all_models_x = self.active_models_x

        if all_models_eta is None:
            all_models_eta = self.active_models_eta

        x = np.sqrt(self.active_noise.temperature.passive/self.k +
                    (self.active_noise.temperature.active /
                     (self.k**2 * self.active_noise.correlation_time + self.k)
                     )
                    ) * torch.randn([self.sample_size, self.sample_dim]).type(torch.DoubleTensor)

        eta = np.sqrt(self.active_noise.temperature.active /
                      self.active_noise.correlation_time) * \
            torch.randn([self.sample_size, self.sample_dim]).type(torch.DoubleTensor)

        with open(self.reverse_x_active_fname, "wb") as f:
            f.write(x.detach().numpy().tobytes())
            
        with open(self.reverse_eta_active_fname, "wb") as f:
            f.write(eta.detach().numpy().tobytes())
            
        time_step_list = []

        if time is None:
            reverse_diffusion_step_start = self.num_diffusion_steps - 2
        else:
            reverse_diffusion_step_start = int(np.ceil(time/self.dt)) - 1

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
                torch.randn(x.shape)

            eta = eta + self.dt*eta/self.active_noise.correlation_time + \
                (2*self.active_noise.temperature.active /
                    (self.active_noise.correlation_time**2)) * Feta*self.dt + \
                (1/self.active_noise.correlation_time) * \
                np.sqrt(2 * self.active_noise.temperature.active * self.dt) * \
                torch.randn(eta.shape)

            with open(self.reverse_x_active_fname, "ab") as f:
                f.write(x.detach().numpy().tobytes())
                
            with open(self.reverse_eta_active_fname, "ab") as f:
                f.write(eta.detach().numpy().tobytes())

        with open(self.reverse_x_active_fname, "r+b") as f:
            mm_r_x = mmap.mmap(f.fileno(), 0)
            mm_r_x_arr = np.frombuffer(mm_r_x, dtype=np.double)
            
            _ = mm_r_x_arr.shape
            
            self.active_reverse_samples_x = \
                torch.from_numpy(np.reshape(mm_r_x_arr, \
                    (len(time_step_list) + 1, self.sample_size, self.sample_dim)))
                
            self.active_reverse_samples_x = [ x for x in self.active_reverse_samples_x]

        with open(self.reverse_eta_active_fname, "r+b") as f:
            mm_r_eta = mmap.mmap(f.fileno(), 0)
            mm_r_eta_arr = np.frombuffer(mm_r_eta, dtype=np.double)
            
            _ = mm_r_eta_arr.shape
            
            self.active_reverse_samples_eta = \
                torch.from_numpy(np.reshape(mm_r_eta_arr, \
                    (len(time_step_list) +1, self.sample_size, self.sample_dim)))
                
            self.active_reverse_samples_eta = [ eta for eta in self.active_reverse_samples_eta]

        self.active_reverse_time_arr = np.array(time_step_list)

        return x.detach(), eta.detach(), self.active_reverse_samples_x, self.active_reverse_samples_eta
