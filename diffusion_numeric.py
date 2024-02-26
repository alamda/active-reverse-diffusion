import abc.ABC

from abc import abstractmethod


class DiffusionNumericAbstract(abc.ABC):
    """Abstract class for diffusion with non-analytic target (starting) dsn"""

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

    @abstractmethod
    def compare_to_target(self):
        """Compare the sample from reverse diffusion to sample from target dsn"""
