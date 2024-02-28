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

    @abstractmethod
    def set_analysis_params(self):
        """Set params necessary for calculating the difference between
        target and calculated dsns"""

    @abstractmethod
    def calc_KL_div(self, data1=None, data2=None):
        """Calculate the KL divergence between two datasets"""

    def diff_method(self, data1=None, data2=None):
        return self.calc_KL_div(data1=data1, data2=data2)
