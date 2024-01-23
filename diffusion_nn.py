from abc import abstractmethod
from diffusion import DiffusionAbstract


class DiffusionNN(DiffusionAbstract):
    """Abstract class for diffusion with non-analytic target (starting) dsn"""

    @abstractmethod
    def nn_architecture(self):
        """Definition of the NN architecture"""

    @abstractmethod
    def loss_fn(self):
        """Definition of loss fn used by NN"""

    @abstractmethod
    def train_nn(self):
        """Train NN on starting data"""

    @abstractmethod
    def run_nn(self):
        """Generate target data using trained NN"""
