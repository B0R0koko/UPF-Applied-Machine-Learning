import json

from solver import Solver

from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def main():
    solver = Solver()

    iterations = list(range(1000))
    wrapped_func = partial(solver.solve_for_n_docks, n_docks=3, sample_prices=True)

    with Pool(processes=10) as p:
        res = list(tqdm(p.imap(wrapped_func, iterations), total=1000))

    with open("result.json", "w") as file:
        json.dump(res, file)


if __name__ == "__main__":
    main()
