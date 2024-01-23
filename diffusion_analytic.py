from abc import abstractmethod
from diffusion import DiffusionAbstract


class DiffusionAnalytic(DiffusionAbstract):
    """Abstract class for diffusion with analytic target (starting) dsn"""

    @abstractmethod
    def starting_dsn(self):
        """Analytic expression for the target (starting) dsn"""

    @abstractmethod
    def score_fn(self):
        """Analytic expression for score function"""

    @abstractmethod
    def calc_kl_divergence(self):
        """Calculate KL divergence between calculated and target dsns"""
