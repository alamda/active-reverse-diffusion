import abc.ABC
from abc import abstractmethod


class TargetAbstract(abc.ABC):
    """Abstract class for the target distribution"""

    @abstractmethod
    def define_target(self):
        """Define the target dsn"""

    @abstractmethod
    def sample(self):
        """Sample the target dsn"""
