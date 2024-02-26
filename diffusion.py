import abc.ABC
from abc import abstractmethod


class DiffusionAbstract(abc.ABC):
    """Abstract class for diffusion"""

    @abstractmethod
    def sample_target(self):
        """Define target (starting) data and sample"""

    @abstractmethod
    def define_noise(self):
        """Define noise objects for the diffusion process"""

    @abstractmethod
    def forward_diffuse(self):
        """Definition of forward diffusion"""

    @abstractmethod
    def reverse_diffuse(self):
        """Uses score from self.calc_score()"""

    @abstractmethod
    def sample_reverse(self):
        """Sample the dsn from reverse diffusion process"""

    @abstractmethod
    def compare_to_target(self):
        """Compare the sample from reverse diffusion to sample from target dsn"""
