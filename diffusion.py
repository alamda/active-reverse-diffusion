from abc import ABC, abstractmethod


class DiffusionAbstract(ABC):
    """Abstract class for diffusion"""

    @abstractmethod
    def generate_target_dsn(self):
        """Arbitrary starting distribution"""

    @abstractmethod
    def forward_process(self):
        """Definition of forward diffusion"""

    @abstractmethod
    def reverse_process(self):
        """Uses score from self.calc_score()"""

    @abstractmethod
    def calc_score(self):
        """Score can be:
            * Estimated through numerics if analytic expr known
                * KL divergence
            * Learned by a NN
                * Loss function
        """

    @abstractmethod
    def diffuse(self):
        """Generic diffuse method
        Calls either self.forward_process() or self.reverse_process()"""
