import abc.ABC
from abc import abstractmethod
from diffusion import DiffusionAbstract


class DiffusionNumericAbstract(DiffusionAbstract):
    """Abstract class for diffusion with non-analytic target (starting) dsn"""

    def __init__(self, name="ndiff"):
        super().__init__(name=name)

    @abstractmethod
    def set_loss_fn(self):
        """Define the loss function to be used for estimating score fn"""

    @abstractmethod
    def set_nn_arch(self):
        """Definition of the NN architecture"""

    @abstractmethod
    def learn_score_fn(self):
        """Train NN on forward diffusion"""
