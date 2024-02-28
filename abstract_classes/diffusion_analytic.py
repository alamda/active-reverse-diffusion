from abc import abstractmethod
from abstract_classes.diffusion import DiffusionAbstract


class DiffusionAnalyticAbstract(DiffusionAbstract):
    """Abstract class for diffusion with analytic target (starting) dsn"""

    def __init__(self, name="adiff", total_time=None, dt=None, num_steps=None,
                 noise_list=None, target=None, dim=None):

        super().__init__(name=name, total_time=total_time, dt=dt, num_steps=num_steps,
                         noise_list=noise_list, target=target, dim=dim)

    @abstractmethod
    def score_fn(self):
        """Define the analytical expression for the score funciton"""

    @abstractmethod
    def initialize_data(self):
        """Initialize the starting dataset"""
