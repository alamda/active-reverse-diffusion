from abc import abstractmethod
from diffusion import DiffusionAbstract


class DiffusionAnalytic(DiffusionAbstract):
    """Abstract class for diffusion with analytic target (starting) dsn"""

    def __init__(self, name="adiff"):
        super().__init__(name=name)

    @abstractmethod
    def set_score_fn(self):
        """Define the analytical expression for the score funciton"""
