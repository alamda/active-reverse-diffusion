from abc import ABC as AbstractBaseClass
from abc import abstractmethod


class TargetAbstract(AbstractBaseClass):
    """Abstract class for the target distribution"""

    def __init__(self, name="target"):
        self.name = name
        self.dsn = None
        self.samples = None

    @abstractmethod
    def define_target(self):
        """Define the target dsn"""

    @abstractmethod
    def sample(self):
        """Sample the target dsn"""
