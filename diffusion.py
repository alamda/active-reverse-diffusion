from abc import ABC, abstractmethod


class DiffusionAbstract(ABC):
    """Abstract class for diffusion"""

    @abstractmethod
    def generate_target_dsn(self):
        """Arbitrary starting distribution"""

    @abstractmethod
    def diffuse_forward(self):
        """Arbitrary forward diffusion process"""

    @abstractmethod
    def calc_score(self):
        """Score can be:
            * Estimated through numerics if analytic expr known
                * KL divergence
            * Learned by a NN
                * Loss function
        """

    @abstractmethod
    def diffuse_backward(self):
        """Uses score from self.calc_score()"""
