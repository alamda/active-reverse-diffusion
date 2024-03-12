from abc import ABC as AbstractBaseClass
from abc import abstractmethod


class Temperature:
    def __init__(self, passive=None, active=None):
        self.set_passive(T=passive)
        self.set_active(T=active)

    def set_passive(self, T=None):
        self.passive = T
        self.check_passive()

    def set_active(self, T=None):
        self.active = T
        self.check_active()

    def check_passive(self):
        try:
            if self.passive is None:
                raise TypeError
        except TypeError:
            print("Passive temperature cannot be None")

    def check_active(self):
        try:
            if self.active is None:
                raise TypeError
        except TypeError:
            print("Active temperature cannot be None")

    def check_all(self):
        self.check_passive()
        self.check_active()


class Noise(AbstractBaseClass):
    def __init__(self, name="noise", dim=None):

        self.name = name
        self.set_dimension(dim=dim)

    @abstractmethod
    def set_temperature(self):
        """Set the temperature for the noise"""

    @abstractmethod
    def check_temperature(self):
        """Check that the temperature value is valid"""

    def set_dimension(self, dim):
        """Set the array dimension for the noise"""
        self.dim = dim
        self.check_dimension()

    def check_dimension(self):
        try:
            if self.dim is None:
                raise TypeError
        except TypeError:
            print("Noise dimension cannot be None")


class NoiseActive(Noise):
    def __init__(self, name="active_noise", Tp=None, Ta=None, tau=None, dim=None):
        super().__init__(name=name, dim=dim)

        self.set_temperature(passive=Tp, active=Ta)
        self.set_correlation_time(tau=tau)

    def set_temperature(self, passive=None, active=None):
        self.temperature = Temperature(passive=passive, active=active)

    def check_temperature(self):
        self.temperature.check_values()

    def set_correlation_time(self, tau=None):
        self.correlation_time = tau
        self.check_correlation_time()

    def check_correlation_time(self):
        try:
            if self.correlation_time is None:
                raise TypeError
        except TypeError:
            print("Correlation time tau cannot be None for active noise")


class NoisePassive(Noise):
    def __init__(self, name="passive_noise", T=None, dim=None):
        super().__init__(name=name, dim=dim)

        self.set_temperature(T=T)

    def set_temperature(self, T=None):
        self.temperature = T
        self.check_temperature()

    def check_temperature(self):
        try:
            if self.temperature is None:
                raise TypeError
        except TypeError:
            print("Temperature cannot be None")


if __name__ == "__main__":
    # Passive noise example
    T = 0.5
    dim = 10000

    myPassiveNoise = NoisePassive(T=T, dim=dim)

    # Active noise example

    Tp = 0.5
    Ta = 0.5
    tau = 0.2
    dim = 10000

    myActiveNoise = NoiseActive(Tp=Tp, Ta=Ta, tau=tau, dim=dim)
