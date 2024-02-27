import abc.ABC
from abc import abstractmethod


class Noise(abc.ABC):
    """Abstract class for Noise"""

    def __init__(self, name="noise", T=None, tau=None):
        self.name = name

        self.set_temperature(T)
        self.set_correlation_time(tau)

        self.current_noise = None

    def set_temperature(self, T):
        """Define temperature (T) for the noise"""
        self.temperature = T

    def set_correlation_time(self, tau):
        """Set the correlation time (tau) for the noise.
            tau = None if noise is passive"""
        self.correlation_time = tau

    @abstractmethod
    def generate_noise(self):
        """Generate an initial value for noise"""

    def get_diffusion_contribution(self):
        """Contribution to the dynamics of the system"""

        try:
            if self.temperature is not None:
                pass
            else:
                raise TypeError
        except TypeError:
            print("noise temperature is None")

        if self.correlation_time is None:  # Passive noise
            self.current_noise = self.generate_noise()
        else:  # Active noise
            if self.current_noise is None:
                self.current_noise = self.generate_noise()

        contribution = self.current_noise

        return contribution

    def evolve(self, dt=None):
        """Propagate noise forward in time (if tau is not None)"""
        if self.correlation_time is not None:
            self.update_noise(dt)

    @abstractmethod
    def update_noise(self, dt=None):
        """Calculate the change in noise and return new value"""