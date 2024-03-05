from abc import ABC as AbstractBaseClass
from abc import abstractmethod

import os
import pickle
from tqdm.auto import tqdm


class DiffusionAbstract(AbstractBaseClass):
    def __init__(self, ofile_base=""):
        self.ofile_base = ofile_base

        self.set_file_names()

        self.load_from_existing_files()

        self.set_target()

        self.run_passive()

        self.run_active()

    def __delete__(self):
        pass

    def load_args(self):
        self.num_diffusion_steps = None
        self.dt = None
        self.passive_temperature = None
        self.active_temperature = None
        self.num_samples = None

        self.num_network_nodes = None
        self.num_network_iterations = None
        pass

    def set_file_names(self):
        self.ofile_target_sample = f"{self.ofile_base}_target_sample.pkl"
        self.pngfile_target_sample = f"{self.ofile_base}_target_hist.png"
        self.histfile_target_sample = f"{self.ofile_base}_target_hist.pkl"

        self.ofile_passive_sample = f"{self.ofile_base}_passive_sample.pkl"
        self.pngfile_passive_sample = f"{self.ofile_base}_passive_sample_hist.png"
        self.histfile_passive_sample = f"{self.ofile_base}_passive_sample_hist.pkl"
        self.ofile_passive_difflist = f"{self.ofile_base}_passive_difflist.pkl"
        self.ofile_passive_model = f"{self.ofile_base}_passive_model.pkl"

        self.ofile_active_sample = f"{self.ofile_base}_active_sample.pkl"
        self.pngfile_active_sample = f"{self.ofile_base}_active_sample_hist.png"
        self.histfile_active_sample = f"{self.ofile_base}_active_sample_hist.pkl"
        self.ofile_active_difflist = f"{self.ofile_base}_active_difflist.pkl"
        self.ofile_active_model = f"{self.ofile_base}_active_model.pkl"

        self.ofile_difflist = f"{self.ofile_base}_difflist_compare.pkl"
        self.pngfile_diff = f"{self.ofile_base}_diff.png"

    def load_file_to_var(self, fname):
        if os.path.isfile(fname):
            print(f"Loading {fname}")

            with open(fname, 'rb') as f:
                var = pickle.load(f)

            return var
        else:
            print(f"File {fname} not found, cannot load. Returning None")
            var = None
            return var

    def load_hist(self, fname):
        if os.path.isfile(fname):
            print(f"Loading {fname}")

            with open(fname, 'rb') as f:
                hist, bins = pickle.load(f)

            return (hist, bins)

        else:
            print(
                f"File {fname} not found, cannot load. Returning (None, None)")
            return (None, None)

    def load_from_existing_files(self):

        # Load samples from existing files
        self.target_sample = self.load_file_to_var(self.ofile_target_sample)
        self.passive_sample = self.load_file_to_var(self.ofile_passive_sample)
        self.active_sample = self.load_file_to_var(self.ofile_active_sample)

        # Load models from existing files
        self.passive_model = self.load_file_to_var(self.ofile_passive_model)
        self.active_model = self.load_file_to_var(self.ofile_active_model)

        # Load hists from existing files
        self.passive_sample_hist, self.passive_sample_bins = \
            self.load_hist(self.histfile_passive_sample)
        self.active_sample_hist, self.active_sample_bins = \
            self.load_hist(self.histfile_active_sample)

        # Load difflists from existing file
        self.passive_difflist = self.load_file_to_var(
            self.ofile_passive_difflist)
        self.active_difflist = self.load_file_to_var(
            self.ofile_active_difflist)
        self.both_difflists = self.load_file_to_var(self.ofile_difflist)

    def save_to_file(self, fname, var):
        with open(fname, 'wb') as f:
            pickle.dump(f, var)
            print(f"{var} saved to file {fname}")

    def set_target(self):
        if self.target_sample is None:
            self.target_sample = self.generate_target()
            self.save_to_file(self.ofile_target_sample, self.target_sample)

    @abstractmethod
    def generate_target(self):
        """Generate a target sample from a distribution"""

    def train_and_sample(self, sample=None, ofile_sample=None,
                         model=None, ofile_model=None,
                         training_fn=None, sampling_fn=None):
        if sample is None:
            if model is None:
                model = training_fn()
                self.save_to_file(ofile_model, model)

            _, sample = sampling_fn(model)
            self.save_to_file(ofile_sample, sample)

    def run_passive(self):
        self.train_and_sample(sample=self.passive_sample,
                              ofile_sample=self.ofile_passive_sample,
                              model=self.passive_model,
                              ofile_model=self.ofile_passive_model,
                              training_fn=self.passive_training,
                              sampling_fn=self.passive_sampling)

    def forward_diffusion_passive(self):
        # Copied from Agnish's code
        distributions, samples = [None], [self.target_sample]
        xt = self.target_sample
        for t in range(self.num_diffusion_steps):
            xt = xt - dt*xt + \
                np.sqrt(2*T*dt)*torch.normal(torch.zeros_like(self.target_sample),
                                             torch.ones_like(self.target_sample))
            samples.append(xt)
        return samples

    def passive_compute_loss(forward_samples, t, score_model):
        # Copied from Agnish's code
        xt = forward_samples[t].type(torch.DoubleTensor)         # x(t)

        l = -(xt - forward_samples[0]*np.exp(-t*self.dt)) / \
            (self.passive_temperature*(1-np.exp(-2*t*self.dt))
             )  # The actual score function

        scr = score_model(xt)
        loss = torch.mean((scr - l)**2)

        return loss, torch.mean(l**2), torch.mean(scr**2)

    def passive_training(self):
        # Copied from Agnish's code
        loss_history = []
        all_models = []
        forward_samples = forward_diffusion_passive(self.target_sample)

        if len(forward_samples[0].shape) == 1:
            num_points = forward_samples[0].shape[0]

            forward_samples = [f.reshape((num_points, 1))
                               for f in forward_samples]

        bar = tqdm(range(1, tsteps))
        t = 1

        for e in bar:
            score_model = torch.nn.Sequential(
                torch.nn.Linear(1, self.num_network_nodes), torch.nn.Tanh(),
                torch.nn.Linear(self.num_network_nodes,
                                self.num_network_nodes), torch.nn.Tanh(),
                torch.nn.Linear(self.num_network_nodes,
                                self.num_network_nodes), torch.nn.Tanh(),
                torch.nn.Linear(self.num_network_nodes, 1)
            ).double()

            optim = torch.optim.AdamW(itertools.chain(
                score_model.parameters()), lr=1e-2, weight_decay=1e-8,)

            loss = 100

            for _ in range(self.num_network_iterations):
                optim.zero_grad()
                loss, l, scr = compute_loss(forward_samples, t, score_model)
                loss.backward()
                optim.step()

            bar.set_description(f'Time:{t} Loss: {loss.item():.4f}')

            all_models.append(copy.deepcopy(score_model))
            loss_history.append(loss.item())

            t = t + 1
        return all_models

    def passive_sampling(self):
        # Copied from Agnish's code

        x = self.passive_temperature * \
            torch.randn(self.num_samples, 1).type(torch.DoubleTensor)

        samples = [x.detach()]

        for t in range(self.num_diffusion_steps-2, 0, -1):
            F = self.passive_model[t](x)

            # If using the total score function
            x = x + x*self.dt + \
                2*self.passive_temperature*F * self.dt + \
                np.sqrt(2*self.passive_temperature*self.dt) * \
                torch.randn(self.num_diffusion_steps, 1)

            samples.append(x.detach())

        return x.detach(), samples

    def run_passive(self):
        self.train_and_sample(sample=self.active_sample,
                              ofile_sample=self.ofile_active_sample,
                              model=self.active_model,
                              ofile_model=self.ofile_active_model,
                              training_fn=self.active_training,
                              sampling_fn=self.active_sampling)

    def forward_diffusion_active(self):
        eta = torch.normal(torch.zeros_like(
            data), np.sqrt(Ta/tau)*torch.ones_like(data))
        samples = [data]
        eta_samples = [eta]
        xt = data
        for t in range(tsteps):
            xt = xt - dt*xt + dt*eta + \
                np.sqrt(2*Tp*dt)*torch.normal(torch.zeros_like(data),
                                              torch.ones_like(data))
            eta = eta - (1/tau)*dt*eta + (1/tau)*np.sqrt(2*Ta*dt) * \
                torch.normal(torch.zeros_like(eta), torch.ones_like(eta))
            samples.append(xt)
            eta_samples.append(eta)
        return samples, eta_samples

    def active_training(self):
        loss_history = []
        all_models_x = []
        all_models_eta = []

        forward_samples, forward_samples_eta = self.forward_diffusion_active()

        if len(forward_samples[0]) == 1:
            num_points = forward_samples[0].shape[0]
            forward_samples = [f.reshape((num_points, 1))
                               for f in forward_samples]

        if len(forward_samples_eta[0]) == 1:
            num_points = forward_samples_eta[0].shape[0]
            forward_samples_eta = [f.reshape((num_points, 1))
                                   for f in forward_samples_eta]
        t = 1
        bar = tqdm(range(1, tsteps))
        for e in bar:
            score_model_x = torch.nn.Sequential(
                torch.nn.Linear(2, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, 1)
            ).double()
            score_model_eta = torch.nn.Sequential(
                torch.nn.Linear(2, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, 1)
            ).double()
            optim_x = torch.optim.AdamW(itertools.chain(
                score_model_x.parameters()), lr=1e-2, weight_decay=1e-8,)
            optim_eta = torch.optim.AdamW(itertools.chain(
                score_model_eta.parameters()), lr=1e-2, weight_decay=1e-8,)
            loss = 100
            for _ in range(iterations):
                optim_x.zero_grad()
                optim_eta.zero_grad()
                loss_x, loss_eta, loss_Fx, loss_Feta, loss_scr_x, loss_scr_eta = score_function_loss(
                    forward_samples, forward_samples_eta, Tp, Ta, tau, k, t, dt, score_model_x, score_model_eta)
                loss_x.backward()
                loss_eta.backward()
                optim_x.step()
                optim_eta.step()
            bar.set_description(
                f'Time:{t} Loss: {loss_x.item():.4f} Fx: {loss_Fx.item():.4f} scr_x: {loss_scr_x.item():.4f}')
            all_models_x.append(copy.deepcopy(score_model_x))
            all_models_eta.append(copy.deepcopy(score_model_eta))
            loss_history.append(loss_x.item())
            t = t + 1
        return all_models_x, all_models_eta

    def active_sampling(self):
        x = np.sqrt(self.passive_temperature/self.k +
                    (self.active_temperature/(self.k**2 * self.active_persistence_time+self.k))) * \
            torch.randn([self.num_samples, 1]).type(torch.DoubleTensor)

        eta = np.sqrt(
            self.active_temperature/self.active_persistence_time)*torch.randn([self.num_samples, 1]).type(torch.DoubleTensor)

        samples_x = [x.detach()]
        samples_eta = [eta.detach()]

        for t in range(self.num_diffusion_steps-2, 0, -1):

            xin = torch.cat((x, eta), dim=1)

            Fx = all_models_x[t](xin)
            Feta = all_models_eta[t](xin)

            x = x + dt*(x-eta) + 2*Tp*Fx*dt + \
                np.sqrt(2*Tp*dt)*torch.randn(x.shape)

            eta = eta + dt*eta/tau + (2*Ta/(tau*tau))*Feta*dt + \
                (1/tau)*np.sqrt(2*Ta*dt)*torch.randn(eta.shape)

            samples_x.append(x.detach())
            samples_eta.append(eta.detach())

        return x.detach(), eta.detach(), samples_x, samples_eta
