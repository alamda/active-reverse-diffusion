from noise import Temperature, NoiseActive, NoisePassive


class TemperatureTest_Factory:
    passive = 1.0
    active = 1.0


def test_temperature_init():
    myFactory = TemperatureTest_Factory()

    myTemperature = Temperature(passive=myFactory.passive,
                                active=myFactory.active)

    assert myTemperature.passive == myFactory.passive
    assert myTemperature.active == myFactory.active


class NoiseActiveTest_Factory:
    Tp = 1.0
    Ta = 1.0
    tau = 0.5
    dim = 100


def test_noise_active_init():
    myFactory = NoiseActiveTest_Factory()

    myActiveNoise = NoiseActive(Tp=myFactory.Tp,
                                Ta=myFactory.Ta,
                                tau=myFactory.tau,
                                dim=myFactory.dim)

    assert myActiveNoise.temperature.passive == myFactory.Tp
    assert myActiveNoise.temperature.active == myFactory.Ta
    assert myActiveNoise.correlation_time == myFactory.tau
    assert myActiveNoise.dim == myFactory.dim


class NoisePassiveTest_Factory:
    T = 1.0
    dim = 100


def test_noise_passive_init():
    myFactory = NoisePassiveTest_Factory()

    myPassiveNoise = NoisePassive(T=myFactory.T,
                                  dim=myFactory.dim)

    assert myPassiveNoise.temperature == myFactory.T
    assert myPassiveNoise.dim == myFactory.dim
