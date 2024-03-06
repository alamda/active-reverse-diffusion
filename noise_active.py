from abstract_classes.noise import NoiseAbstract
import numpy as np


class NoiseActive(NoiseAbstract):
    def __init__(self, name="passive_noise", T=None, tau=None, dim=None):
        super().__init__(name=name, T=T, tau=tau, dim=dim)

        self.check_correlation_time()

    def initialize_noise(self):
        pass

    def check_correlation_time(self):
        try:
            if self.correlation_time is None:
                raise TypeError
        except TypeError:
            print("Correlation time tau cannot be None for active noise")

    def generate_noise(self, dt=None):
        return (1/self.correlation_time) * \
            np.sqrt(2*self.temperature*dt) * \
            np.random.randn(self.dim)

    def update(self, dt=None, score=None):
        if self.current_noise is None:
            self.current_noise = self.generate_noise(dt=dt)

        self.current_noise += -(self.current_noise/self.correlation_time)*dt + \
            (2*self.temperature / (self.correlation_time**2))*score*dt + \
            self.generate_noise(dt=dt)


if __name__ == "__main__":
    T = 0.5
    tau = 0.2
    dim = 10000
    dt = 0.02

    myActiveNoise = NoiseActive(T=T, tau=tau, dim=dim)

    myActiveNoise.update(dt=dt)
