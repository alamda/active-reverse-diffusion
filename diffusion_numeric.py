import abc.ABC
from abc import abstractmethod
from diffusion import DiffusionAbstract


class DiffusionNumericAbstract(DiffusionAbstract):
    """Abstract class for diffusion with non-analytic target (starting) dsn"""

    def __init__(self, name="ndiff"):
        super().__init__(name=name)

        self.loss_fn = None
        self.nn_arch = None
        self.score_fn = None

    @abstractmethod
    def set_loss_fn(self):
        """Define the loss function to be used for estimating score fn"""

    @abstractmethod
    def set_nn_arch(self):
        """Definition of the NN architecture"""

    @abstractmethod
    def learn_score_fn(self):
        """Train NN on forward diffusion"""

    @abstractmethod
    def reverse_diffuse(self):
        """Uses score function to generate new dsn"""

    @abstractmethod
    def sample_reverse(self):
        """Generate target data from result of reverse diffusion"""
