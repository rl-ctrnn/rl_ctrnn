"""
To run this module, execute `python -m cooper.demo.sweep_weights`
    from this repository's base directory.

To save the outputs to a file, use stdout redirection:
    `python -m cooper.demo.sweep_weights > data.csv`
"""

from typing import List
from typing_extensions import TypeAlias
from multiprocessing import Pool

from cooper.ctrnn import Ctrnn
from cooper.util import get_beers_fitness

Weight: TypeAlias = tuple[int, int]
Datum: TypeAlias = tuple[float, float, float]
Param: TypeAlias = tuple[float, float]

THREAD_COUNT = 10
BOUNDS = 16.0
STEP = 0.25
PROGENITOR = Ctrnn()
WEIGHT_A = (0, 1)
WEIGHT_B = (1, 0)
LOG_OUTPUT = True


def get_sweep(step: float = STEP, bounds: float = BOUNDS) -> List[Param]:
    params: List[Param] = []
    y = -bounds
    while y <= BOUNDS:
        x = -bounds
        while x <= BOUNDS:
            params.append((x, y))
            x += step
        y += step
    return params


def main(param: Param) -> Datum:
    ctrnn = PROGENITOR.clone()
    ctrnn.weights[WEIGHT_A] = param[0]
    ctrnn.weights[WEIGHT_B] = param[1]
    fitness = get_beers_fitness(ctrnn)
    datum = (param[0], param[1], fitness)
    if LOG_OUTPUT:
        print(f"{datum[0]},{datum[1]},{datum[2]}")
    return datum


def to_csv(data: List[Datum]) -> str:
    header = "a,b,fitness\n"
    lines = [f"{d[0]},{d[1]},{d[2]}" for d in data]
    return header + "\n".join(lines)


if __name__ == "__main__":
    print("a,b,fitness")
    p = Pool(THREAD_COUNT)
    sweep = get_sweep()
    data = p.map(main, sweep)
