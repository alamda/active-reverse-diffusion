

class DiffusionNumeric:
    def __init__(self, ofile_base="", passive_noise=None, active_noise=None, target=None,
                 num_diffusion_steps=None, dt=None, k=1, sample_dim=None):
        self.ofile_base = ofile_base

        self.passive_noise = passive_noise
        self.active_noise = active_noise

        self.target = target

        self.num_diffusion_steps = num_diffusion_steps
        self.dt = dt

        self.k = k

        self.sample_dim = sample_dim

    def forward_diffusion_passive(self):
        distributions, samples = [None], [target_sample]
        xt = target_sample
        for t in range(tsteps):
            xt = xt - self.dt*xt + np.sqrt(2*self.passive_noise.temperature*self.dt) * torch.normal(
                torch.zeros_like(self.target.sample), torch.ones_like(self.target.sample))
            samples.append(xt)
        return samples

    def compute_loss_passive(self, forward_samples, t, score_model):
        xt = forward_samples[t].type(torch.DoubleTensor)         # x(t)
        l = -(xt - forward_samples[0]*np.exp(-t*self.dt)) / \
            (self.passive_noise.temperature *
             (1-np.exp(-2*t*self.dt)))  # The actual score function
        scr = score_model(xt)
        loss = torch.mean((scr - l)**2)
        return loss, torch.mean(l**2), torch.mean(scr**2)

    def train_diffusion_passive(self, dataset, nrnodes=4, iterations=500):
        loss_history = []
        all_models = []
        forward_samples = do_diffusion(
            dataset, self.num_diffusion_steps, self.passive_noise.temperature, self.dt)

        if len(forward_samples[0].shape) == 1:
            num_points = forward_samples[0].shape[0]

            forward_samples = [f.reshape((self.sample_dim, 1))
                               for f in forward_samples]
        bar = tqdm(range(1, self.num_diffusion_steps))

        t = 1

        for e in bar:
            score_model = torch.nn.Sequential(
                torch.nn.Linear(1, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
                torch.nn.Linear(nrnodes, 1)
            ).double()

            optim = torch.optim.AdamW(itertools.chain(
                score_model.parameters()), lr=1e-2, weight_decay=1e-8,)

            loss = 100

            for _ in range(iterations):
                optim.zero_grad()
                loss, l, scr = compute_loss(
                    forward_samples, self.passive_noise.temperature, t, self.dt, score_model)

                loss.backward()
                optim.step()

            bar.set_description(f'Time:{t} Loss: {loss.item():.4f}')

            all_models.append(copy.deepcopy(score_model))

            loss_history.append(loss.item())

            t = t + 1

        return all_models

    def sample_from_diffusion_passive(self, all_models):
        x = self.passive_noise.temperature * \
            torch.randn(self.sample_dim, 1).type(torch.DoubleTensor)

        samples = [x.detach()]

        for t in range(self.num_diffusion_steps-2, 0, -1):

            F = all_models[t](x)
            # If using the total score function

            x = x + x*self.dt + 2*self.passive_noise.temperature*F*self.dt + \
                np.sqrt(2*self.passive_noise.temperature*self.dt) * \
                torch.randn(self.sample_dim, 1)

            samples.append(x.detach())

        return x.detach(), samples

    def forward_diffusion_active(data):
        eta = torch.normal(torch.zeros_like(data),
                           np.sqrt(self.active_noise.temperature /
                                   self.active_noise.correlation_time)
                           * torch.ones_like(data)
                           )
        samples = [data]
        eta_samples = [eta]
        xt = data

        for t in range(self.num_diffusion_steps):
            xt = xt - self.dt*xt + self.dt*eta + \
                np.sqrt(2*self.passive_noise.temperature*self.dt) * \
                torch.normal(torch.zeros_like(data), torch.ones_like(data))

            eta = eta - (1/self.active_noise.correlation_time)*self.dt*eta + \
                (1/self.active_noise.correlation_time) * \
                np.sqrt(2*self.active_noise.temperature*self.dt) * \
                torch.normal(torch.zeros_like(eta), torch.ones_like(eta))

            samples.append(xt)
            eta_samples.append(eta)

        return samples, eta_samples

    def compute_loss_active(self, t,
                            forward_samples, forward_samples_eta,
                            score_model_x, score_model_eta):

        a = np.exp(-self.k*t*self.dt)

        b = (np.exp(-t*self.dt/self.active_noise.correlation_time) -
             np.exp(-self.k*t*self.dt))/(self.k-(1/self.active_noise.correlation_time))

        c = np.exp(-t*self.dt/self.active_noise.correlation_time)

        M11, M12, M22 = M_11_12_22(t*self.dt)

        det = M11*M22 - M12*M12
        x0 = forward_samples[0]
        eta0 = forward_samples_eta[0]
        x = forward_samples[t].type(torch.DoubleTensor)
        eta = forward_samples_eta[t].type(torch.DoubleTensor)

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

    def M_11_12_22(t):

        a = np.exp(-self.k*t)
        b = np.exp(-t/self.active_noise.correlation_time)

        Tx = self.passive_noise.temperature
        Ty = self.active_noise.temperature / \
            (self.active_noise.correlation_time**2)
        w = (1/self.active_noise.correlation_time)

        M11 = (1/self.k)*Tx*(1-a**2) + (1/self.k)*Ty*(1/(w*(self.k+w)) + 4*a*b *
                                                      self.k/((self.k+w)*(self.k-w)**2) - (self.k*b**2 + w*a**2)/(w*(self.k-w)**2))
        M12 = (Ty/(w*(self.k**2 - w**2))) * (k*(1-b**2) - w*(1 + b**2 - 2*a*b))
        M22 = (Ty/w)*(1-b**2)

        return M11, M12, M22

    def train_diffusion_active(self):
        loss_history = []
        all_models_x = []
        all_models_eta = []
        forward_samples, forward_samples_eta = do_diffusion_active(
            dataset, tsteps, Tp, Ta, tau, dt)

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

    def sample_from_diffusion_active(self):
        x = np.sqrt(Tp/k + (Ta/(k*k*tau+k))) * \
            torch.randn([N, 1]).type(torch.DoubleTensor)

        eta = np.sqrt(Ta/tau)*torch.randn([N, 1]).type(torch.DoubleTensor)

        samples_x = [x.detach()]
        samples_eta = [eta.detach()]

        for t in range(tsteps-2, 0, -1):
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
