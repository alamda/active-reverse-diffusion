from abc import abstractmethod
from diffusion import DiffusionAbstract


class DiffusionAnalytic(DiffusionAbstract):
    """Abstract class for diffusion with analytic target (starting) dsn"""

    @abstractmethod
    def define_score_function(self):
        """Define the analytical expression for the score funciton"""
