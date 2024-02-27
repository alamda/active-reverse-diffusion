from abstract_classes.noise import NoiseAbstract
import numpy as np


class NoisePassive(NoiseAbstract):
    def __init__(self, name="passive_noise", T=None, tau=None):
        super().__init__(name=name, T=T, tau=tau)

    def generate_noise(self, dt=None, dim=None):
        return np.sqrt(2*self.temperature*dt) * np.random.randn(dim)

    def update(self, dt=None, dim=None):
        self.current_noise = self.generate_noise(dt=dt, dim=dim)


if __name__ == "__main__":
    myPassiveNoise = NoisePassive(T=0.5)

    print(myPassiveNoise.generate_noise(dt=0.1, dim=5))

    del myPassiveNoise
