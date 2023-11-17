"""Run the simulation for a simple two-airport network."""
from copy import deepcopy

import pandas as pd
import pyro
import torch

from bayes_air.model import air_traffic_network_model
from bayes_air.network import NetworkState
from bayes_air.schedule import parse_schedule


def main():
    pyro.enable_validation(True)
    pyro.set_rng_seed(1)

    # Define a simple schedule that sends the aircraft from airport 0 to airport 1 and back
    # Time is defined in hours from the start of the simulation
    states = []
    for _ in range(1):
        schedule = pd.DataFrame(
            {
                "flight_number": ["F1", "F1"],
                "origin_airport": ["A1", "A2"],
                "destination_airport": ["A2", "A1"],
                "scheduled_departure_time": [0.0, 2.0],
                "scheduled_arrival_time": [1.0, 3.5],
                "actual_departure_time": [
                    0.0 + torch.rand(()) * 0.1,
                    2.0 + torch.rand(()) * 0.1,
                ],
                "actual_arrival_time": [
                    1.0 + torch.rand(()) * 0.1,
                    3.5 + torch.rand(()) * 0.1,
                ],
            }
        )
        flights, airports = parse_schedule(schedule)
        airports[0].available_aircraft.append(torch.tensor(0.0))
        airports[0].available_crew.append(torch.tensor(0.0))
        state = NetworkState(
            airports={airport.code: airport for airport in airports},
            pending_flights=flights,
        )
        states.append(state)

    hrs = 24.0
    dt = 0.1

    model_graph = pyro.render_model(
        air_traffic_network_model,
        model_args=(deepcopy(states), hrs, dt),
        render_params=False,
        render_distributions=False,
    )
    model_graph.render("tmp/two_airport_network", cleanup=True)

    # # Print dependencies
    # print("Dependencies:")
    # pprint.pprint(
    #     pyro.infer.inspect.get_dependencies(
    #         air_traffic_network_model, (deepcopy(states), hrs, dt)
    #     )
    # )

    # Set up MCMC inference
    num_samples = 50
    warmup_steps = 50
    num_chains = 4
    model = air_traffic_network_model
    nuts_kernel = pyro.infer.NUTS(
        model,
        max_tree_depth=5,
        jit_compile=False,
        step_size=3e-2,
        adapt_step_size=False,
    )
    mcmc = pyro.infer.MCMC(
        nuts_kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
    )
    mcmc.run(deepcopy(states), hrs, dt)
    mcmc.summary(prob=0.9)


if __name__ == "__main__":
    main()
