import pulp
import pandas as pd
import numpy as np
import os
import itertools

from functools import partial
from typing import *


class Solver:
    # define dictionary of additional fees for starting a contract with dock
    ADDITIONAL_COSTS = {
        "Algeciras": 800_000,
        "Marseille": 500_000,
        "Antwerp": 1_000_000,
    }

    RANDOM_PRICES = {
        "Valencia": partial(np.random.uniform, 310, 390),
        "Algeciras": partial(np.random.uniform, 280, 330),
    }

    def __init__(self) -> Self:
        self.load_prices()

    def load_prices(self) -> None:
        self.df = pd.read_csv("prices_and_capacities.csv")

    def get_prices(
        self, docks: List[str], sample_prices: bool
    ) -> List[Union[int, float]]:
        if not sample_prices:
            return self.df[self.df.dock.isin(docks)]["40_ft_container_price_eur"].values

        prices = []
        for dock in docks:
            if dock in self.RANDOM_PRICES:
                prices.append(self.RANDOM_PRICES[dock]())
            else:
                prices.append(
                    self.df.loc[self.df.dock == dock, "40_ft_container_price_eur"].iloc[
                        0
                    ]
                )
        return prices

    def solve_problem(
        self, docks: List[str], prices: List[str], max_capacities: List[str]
    ):
        problem = pulp.LpProblem("cost_minimisation", pulp.LpMinimize)

        quantities = [
            pulp.LpVariable(dock, lowBound=0, cat=pulp.LpContinuous) for dock in docks
        ]
        obj_func = pulp.lpSum([q * p for q, p in zip(quantities, prices)])

        # add additional one time fixed costs for starting a contract
        for dock in docks:
            if dock in self.ADDITIONAL_COSTS:
                obj_func += self.ADDITIONAL_COSTS[dock]

        problem += obj_func

        for quantity, max_capacity in zip(quantities, max_capacities):
            problem += quantity <= max_capacity

        problem += pulp.lpSum([q * 66 * 0.45 for q in quantities]) == 1_500_000

        problem.solve(pulp.PULP_CBC_CMD(msg=False))

        return {
            "status": problem.status,
            "variables": {var.name: var.value() for var in problem.variables()},
            "objective_value": problem.objective.value(),
        }

    def solve_for_n_docks(
        self, process_id=None, n_docks=3, sample_prices=False
    ) -> Dict[str, float]:
        min_cost, best_solution = None, None

        for combination in itertools.combinations(self.df.dock, n_docks):
            prices = self.get_prices(combination, sample_prices)
            max_capacities = self.df[self.df.dock.isin(combination)][
                "max_capacity"
            ].values

            res = self.solve_problem(
                docks=combination, prices=prices, max_capacities=max_capacities
            )

            if res["status"] == -1:
                continue

            if not min_cost:
                min_cost, best_solution = res["objective_value"], res
                continue

            if res["objective_value"] < min_cost:
                min_cost, best_solution = res["objective_value"], res

        return {**best_solution["variables"], "objective_value": min_cost}


if __name__ == "__main__":
    solver = Solver()
    print(solver.solve_for_n_docks(n_docks=3, sample_prices=True))
