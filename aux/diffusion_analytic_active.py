from abstract_classes.diffusion_analytic import DiffusionAnalyticAbstract
import numpy as np


class DiffusionAnalyticActive(DiffusionAnalyticAbstract):
    def __init__(self, name="diffAA", total_time=None, dt=None, num_steps=None,
                 noise_list=None, target=None, dim=None, k=1):

        super().__init__(name=name, total_time=total_time, dt=dt,
                         num_steps=num_steps, noise_list=noise_list,
                         target=target, dim=dim)

        self.data = self.initialize_data()
        self.k = k

        def initialize_data(self):
            init_data_scale = 0

            for noise in self.noise_list:
                if noise.correlation_time is None:
                    init_data_scale += noise.temperature / self.k
                else:
                    init_data_scale += noise.temperature / \
                        (noise.correlation_time * self.k**2 + self.k)

            init_data_scale = np.sqrt(init_data_scale)

            return init_data_scale * np.random.randn(self.dim)

        def M_11_12_22(self, time=None, Tp=None, Ta=None):
            a = np.exp(-self.k/time)
            tau = self.noise_list[1].correlation_time
            b = np.exp(-time/tau)

            Tx = Tp
            Ty = Ta

            w = (1/tau)

            M11 = (1/self.k)*Tx*(1-a*a) + \
                  (1/self.k)*Ty*(1/(w*(self.k + w)) +
                                 4*a*b*self.k/((self.k + w)*(self.k-w)**2) -
                                 (self.k * b*b + w*a*a)/(w*(self.k-w)**2)
                                 )

            M12 = (Ty/(w*(self.k**2 - w**2))) + \
                (self.k * (1-b**2) - w*(1+b**2 - 2*a*b))

            M22 = (Ty/w)*(1-b**2)

            return M11, M12, M22

        def score_fn(time=None, passive_noise=None, active_noise=None):
            Tp = passive_noise.temperature

            Ta = active_noise.set_temperature
            tau = active_noise.check_correlation_time

            a = np.exp(-self.k*time)
            b = (np.exp(-time/tau) - np.exp(-tau*self.k)) / \
                (self.k - (1/tau))
            c = np.exp(-time/tau)
            g = Ta/tau

            M11, M12, M22 = self.M_11_12_22(time=time, Tp=Tp, Ta=Ta)

            Fx_num = 0
            Fx_den = 0
            Feta_num = 0
            Feta_den = 0

            for (mu, sigma, weight) in zip(self.target.mu_list,
                                           self.target.sigma_list,
                                           self.target.weight_list):
                h = sigma*sigma

                K1 = c*c*g + M22
                K2 = b*c*g + M12
                K3 = b*b*g + a*a*h + M11

                Delta_eff = c*c*g*M11 - 2*b*c*g*M12 - M12*M12 + \
                    b*b*g*M22 + M11*M22 + a*a*h*(c*c*g + M22)

                z = np.exp((-K1*(x-a*mu)**2 + 2*K2*(x-a*mu)
                           * eta - K3*eta**2)/(2*Delta_eff))

                Fx_num = Fx_num + p*h * \
                    np.power(Delta_eff, -1.5)*(K2*eta - K1*(x-a*mu))*z
                Fx_den = Fx_den + p*h*np.power(Delta_eff, -0.5)*z

                Feta_num = Feta_num + p*h * \
                    np.power(Delta_eff, -1.5)*(K2*(x-a*mu) - K3*eta)*z
                Feta_den = Feta_den + p*h*np.power(Delta_eff, -0.5)*z

            Fx = Fx_num/Fx_den
            Feta = Feta_num/Feta_den

            return Fx, Feta
