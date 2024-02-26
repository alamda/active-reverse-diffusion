from abc import abstractmethod
from diffusion import DiffusionAbstract


class DiffusionAnalytic(DiffusionAbstract):
    """Abstract class for diffusion with analytic target (starting) dsn"""

    @abstractmethod
    def add_target(self):
        """Set target data"""

    @abstractmethod
    def add_noise(self):
        """Define noise objects for the diffusion process"""

    @abstractmethod
    def forward_diffuse(self):
        """Definition of forward diffusion"""

    @abstractmethod
    def set_score_fn(self):
        """Define the analytical expression for the score funciton"""

    @abstractmethod
    def reverse_diffuse(self):
        """Uses score function"""

    @abstractmethod
    def sample_reverse(self):
        """Generate target data from result of reverse diffusion"""

    @abstractmethod
    def compare_to_target(self):
        """Compare the sample from reverse diffusion to sample from target dsn"""
