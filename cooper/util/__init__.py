from cooper.behavior.oscillator import Oscillator
from cooper.ctrnn import Ctrnn


def get_beers_fitness(ctrnn: Ctrnn, duration: float = 300, window: float = 50) -> float:
    voltages = ctrnn.init_voltage()
    behavior = Oscillator(dt=0.01, size=ctrnn.size, duration=duration, window=window)
    behavior.setup(ctrnn.get_output(voltages))
    while behavior.time < behavior.duration:
        voltages = ctrnn.step(0.01, voltages)
        behavior.grade(ctrnn.get_output(voltages))
    return behavior.fitness
