from abc import ABC, abstractmethod


class DiffusionAbstract(ABC):
    """Abstract class for diffusion"""

    @asbtractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __delete__(self):
        pass

    @abstractmethod
    def generate_target_dsn(self):
        """Arbitrary starting distribution"""
        pass

    @abstractmethod
    def diffuse_forward(self):
        """Arbitrary forward diffusion process"""
        pass

    @abstractmethod
    def calc_score(self):
        """Score can be:
            * Estimated through numerics if analytic expr known
                * KL divergence
            * Learned by a NN
                * Loss function
        """
        pass

    @abstractmethod
    def diffuse_backward(self):
        """Uses score from self.calc_score()"""
        pass
