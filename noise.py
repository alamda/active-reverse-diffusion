import abc.ABC
from abc import abstractmethod


class Noise(abc.ABC):
    """Abstract class for Noise"""

    @abstractmethod
    def set_temperature(self):
        """Define temperature (T) for the noise"""

    @abstractmethod
    def set_correlation(self):
        """Set the correlation time (tau) for the noise.
            tau = None if noise is passive"""

    @abstractmethod
    def generate_noise(self):
        """Generate an initial value for noise"""

    @abstractmethod
    def evolve(self):
        """Propagate noise forward in time (if tau is not None)"""
