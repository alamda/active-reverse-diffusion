from diffusion import Diffusion

import numpy as np


class DiffusionAnalytic(Diffusion):
    def __init__(self, ofile_base="", passive_noise=None, active_noise=None, target=None,
                 num_diffusion_steps=None, dt=None, k=1, sample_dim=None, data_proc=None):

        super().__init__(ofile_base=ofile_base,
                         passive_noise=passive_noise,
                         active_noise=active_noise,
                         target=target,
                         num_diffusion_steps=num_diffusion_steps,
                         dt=dt,
                         k=k,
                         sample_dim=sample_dim,
                         data_proc=data_proc,
                         diffusion_type='analytic')

    def score_function_passive(self, x=None, t=None):

        try:
            if self.target.mu_list is not None:
                a = np.exp(-t)

                Delta = self.passive_noise.temperature*(1-np.exp(-2*t))
                Fx_num = 0
                Fx_den = 0

                for mu, sigma, pi in zip(self.target.mu_list, self.target.sigma_list, self.target.pi_list):
                    h = sigma*sigma

                    Delta_eff = a*a*h + Delta

                    z = np.exp((-(x-a*mu)**2)/(2*Delta_eff))

                    Fx_num -= pi * \
                        np.sqrt(h) * np.power(Delta_eff, -1.5) * (x - a*mu) * z

                    Fx_den += pi * np.sqrt(h) * np.power(Delta_eff, -0.5) * z

                Fx = Fx_num/Fx_den

                return Fx
            else:
                raise TypeError
        except TypeError:
            print("Analytical diffusion implemented only for 'gaussian' target type")

    def sample_from_diffusion_passive(self, time=None):
        x_t = np.sqrt(self.passive_noise.temperature) * \
            np.random.randn(self.sample_dim)

        samples = [x_t]

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

            F = self.score_function_passive(x=x_t, t=time_now)

            x_t = x_t + x_t*self.dt + 2*self.passive_noise.temperature*F*self.dt + \
                np.sqrt(2*self.passive_noise.temperature*self.dt) * \
                np.random.randn(len(x_t))

            samples.append(x_t)

        self.passive_reverse_time_arr = np.array(time_step_list)

        self.passive_reverse_samples = samples

        return x_t, samples

    def M_11_12_22(self, t=None):

        a = np.exp(-self.k*t)

        b = np.exp(-t/self.active_noise.correlation_time)

        Tx = self.active_noise.temperature.passive

        Ty = self.active_noise.temperature.active / \
            (self.active_noise.correlation_time**2)

        w = (1/self.active_noise.correlation_time)

        M11 = (1/self.k)*Tx*(1-a*a) + (1/self.k)*Ty*(1/(w*(self.k+w)) + 4*a*b *
                                                     self.k/((self.k+w)*(self.k-w)**2) - (self.k*b**2 + w*a*a)/(w*(self.k-w)**2))
        M12 = (Ty/(w*(self.k**2 - w*w))) * \
            (self.k*(1-b*b) - w*(1 + b*b - 2*a*b))

        M22 = (Ty/w)*(1-b*b)

        return M11, M12, M22

    def score_function_active(self, x=None, eta=None, t=None):
        try:
            if self.target.mu_list is not None:  # in ("gaussian", "Gaussian"):
                a = np.exp(-self.k*t)

                b = (np.exp(-t/self.active_noise.correlation_time) -
                     np.exp(-self.k*t))/(self.k-(1/self.active_noise.correlation_time))

                c = np.exp(-t/self.active_noise.correlation_time)

                g = self.active_noise.temperature.active / self.active_noise.correlation_time

                M11, M12, M22 = self.M_11_12_22(t=t)

                Fx_num = 0
                Fx_den = 0
                Feta_num = 0
                Feta_den = 0

                for mu, sigma, pi in zip(self.target.mu_list, self.target.sigma_list, self.target.pi_list):
                    h = sigma*sigma

                    K1 = c*c*g + M22

                    K2 = b*c*g + M12

                    K3 = b*b*g + a*a*h + M11

                    Delta_eff = c*c*g*M11 - 2*b*c*g*M12 - M12*M12 + \
                        b*b*g*M22 + M11*M22 + a*a*h*(c*c*g + M22)

                    z = np.exp((-K1*(x-a*mu)**2 + 2*K2*(x-a*mu)
                                * eta - K3*eta**2)/(2*Delta_eff))

                    Fx_num = Fx_num + pi*h * \
                        np.power(Delta_eff, -1.5)*(K2*eta - K1*(x-a*mu))*z

                    Fx_den = Fx_den + pi*h*np.power(Delta_eff, -0.5)*z

                    Feta_num = Feta_num + pi*h * \
                        np.power(Delta_eff, -1.5)*(K2*(x-a*mu) - K3*eta)*z

                    Feta_den = Feta_den + pi*h*np.power(Delta_eff, -0.5)*z

                Fx = Fx_num/Fx_den

                Feta = Feta_num/Feta_den

            else:
                Fx = None
                Feta = None
                raise TypeError

            return Fx, Feta
        except TypeError:
            print("Analytical diffusion implemented only for 'gaussian' target type")

    def sample_from_diffusion_active(self, time=None):
        x = np.sqrt(self.active_noise.temperature.passive/self.k +
                    (self.active_noise.temperature.active /
                     (self.k**2 * self.active_noise.correlation_time + self.k)
                     )
                    ) * np.random.randn(self.sample_dim)

        eta = np.sqrt(self.active_noise.temperature.active /
                      self.active_noise.correlation_time) * \
            np.random.randn(self.sample_dim)

        samples_x = [x]
        samples_eta = [eta]

        time_step_list = []

        if time is None:
            reverse_diffusion_step_start = self.num_diffusion_steps - 2
        else:
            reverse_diffusion_step_start = int(np.ceil(time/self.dt)) - 1

        self.num_active_reverse_diffusion_steps = reverse_diffusion_step_start + 1

        for t in range(reverse_diffusion_step_start, 0, -1):

            time_now = t*self.dt
            time_step_list.append(time_now)

            Fx, Feta = self.score_function_active(x=x, eta=eta, t=time_now)

            x = x + self.dt*(x-eta) + \
                2*self.active_noise.temperature.passive*Fx*self.dt + \
                np.sqrt(2*self.active_noise.temperature.passive*self.dt) * \
                np.random.randn(self.sample_dim)

            eta = eta + self.dt*eta/self.active_noise.correlation_time + \
                (2*self.active_noise.temperature.active /
                    (self.active_noise.correlation_time**2)) * Feta*self.dt + \
                (1/self.active_noise.correlation_time) * \
                np.sqrt(2 * self.active_noise.temperature.active * self.dt) * \
                np.random.randn(self.sample_dim)

            samples_x.append(x)
            samples_eta.append(eta)

        self.active_reverse_time_arr = np.array(time_step_list)

        self.active_reverse_samples_x = samples_x
        self.active_reverse_samples_eta = samples_eta

        return x, eta, samples_x, samples_eta
