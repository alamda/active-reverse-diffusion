from abc import abstractmethod
from diffusion import DiffusionAbstract


class DiffusionNN(DiffusionAbstract):
    """Abstract class for diffusion with non-analytic target (starting) dsn"""

    @abstractmethod
    def define_loss_function(self):
        """Define the loss function to be used for estimating score fn"""

    @abstractmethod
    def define_nn_architecture(self):
        """Definition of the NN architecture"""

    @abstractmethod
    def train_nn(self):
        """Train NN on starting data"""

    @abstractmethod
    def run_nn(self):
        """Generate target data using trained NN"""
