from abc import ABC as AbstractBaseClass
from abc import abstractmethod


class NoiseAbstract(AbstractBaseClass):
    """Abstract class for Noise"""

    def __init__(self, name="noise", T=None, tau=None, dim=None):
        self.name = name

        self.set_temperature(T)
        self.set_correlation_time(tau)
        self.set_dimension(dim)

        self.check_temperature()
        self.check_dimension()

        self.current_noise = None

        self.noise_list = []

    def set_temperature(self, T):
        """Define temperature (T) for the noise"""
        self.temperature = T

    def set_correlation_time(self, tau):
        """Set the correlation time (tau) for the noise.
            tau = None if noise is passive"""
        self.correlation_time = tau

    def set_dimension(self, dim):
        """Set the array dimension for the noise"""
        self.dim = dim

    def check_temperature(self):
        try:
            if self.temperature is None:
                raise TypeError
        except TypeError:
            print("Noise temperature (T) cannot be None")

    def check_dimension(self):
        try:
            if self.dim is None:
                raise TypeError
        except TypeError:
            print("Noise dimension (dim) cannot be None")

    @ abstractmethod
    def generate_noise(self):
        """Generate random value for noise"""

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

    @ abstractmethod
    def update(self):
        """Propagate noise forward in time (if tau is not None)"""

    def record(self, time=None):
        try:
            if time is not None:
                self.noise_list.append([time, self.current_noise])
            else:
                raise TypeError
        except TypeError:
            print("Time cannot be None")
