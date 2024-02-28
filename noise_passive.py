from abstract_classes.noise import NoiseAbstract
import numpy as np


class NoisePassive(NoiseAbstract):
    def __init__(self, name="passive_noise", T=None, tau=None, dim=None):
        super().__init__(name=name, T=T, tau=tau, dim=dim)

    def initialize_noise(self, dt=None):
        self.current_noise = self.generate_noise(dt=dt)

    def generate_noise(self, dt=None):
        return np.sqrt(2*self.temperature*dt) * np.random.randn(self.dim)

    def update(self, dt=None, score=None):
        self.current_noise = self.generate_noise(dt=dt)


if __name__ == "__main__":
    dim = 10000
    dt = 0.02

    myPassiveNoise = NoisePassive(T=0.5, dim=dim)
    myPassiveNoise.initialize_noise(dt=dt)

    print(myPassiveNoise.generate_noise(dt=dt))

    del myPassiveNoise
