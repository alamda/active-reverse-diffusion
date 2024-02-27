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

    @abstractmethod
    def initialize_noise(self):
        """Generate an initial value for the noise"""

    @ abstractmethod
    def generate_noise(self, dt=None):
        """Generate random value for noise"""

    def get_diffusion_contribution(self, dt=None):
        """Contribution to the dynamics of the system"""

        if self.correlation_time is None:  # Passive noise
            self.current_noise = self.generate_noise(dt=dt)
        else:  # Active noise
            if self.current_noise is None:
                self.current_noise = self.generate_noise(dt=dt)

        contribution = self.current_noise

        return contribution

    @ abstractmethod
    def update(self):
        """Propagate noise forward in time (if tau is not None)
        or generate a new random sample of noise (if tau is None)"""
