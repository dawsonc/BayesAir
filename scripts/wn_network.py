"""Run the simulation for a simple two-airport network."""
import pyro
import torch

import bayes_air.utils.dataloader as ba_dataloader
from bayes_air.model import air_traffic_network_model
from bayes_air.network import NetworkState
from bayes_air.schedule import parse_schedule


def train_nominal():
    pyro.enable_validation(True)
    pyro.set_rng_seed(1)

    # Set the number of starting aircraft and crews at each airport
    starting_crew = 50
    starting_aircraft = 50

    # Hyperparameters
    hrs = 25.0  # extend a bit past 1 day to let all flights finish
    dt = 0.25
    top_N = 2

    num_samples = 100
    warmup_steps = 100
    num_chains = 1

    # Load the dataset
    df = ba_dataloader.load_all_data()
    df = ba_dataloader.remap_columns(df)
    df = ba_dataloader.top_N_df(df, top_N)
    nominal_df, _ = ba_dataloader.split_nominal_disrupted_data(df)
    nominal_dfs = ba_dataloader.split_by_date(nominal_df)

    # Convert each day into a schedule
    states = []
    for day_df in nominal_dfs:
        flights, airports = parse_schedule(day_df)
        for _ in range(starting_aircraft):
            airports[0].available_aircraft.append(torch.tensor(0.0))
        for _ in range(starting_crew):
            airports[0].available_crew.append(torch.tensor(0.0))
        state = NetworkState(
            airports={airport.code: airport for airport in airports},
            pending_flights=flights,
        )
        states.append(state)

    # Create an autoguide for the model
    model = air_traffic_network_model

    # Render the first few hours of the model to see what the structure is
    model_graph = pyro.render_model(
        air_traffic_network_model,
        model_args=(states, hrs, dt),
        render_params=False,
        render_distributions=False,
    )
    model_graph.render(f"tmp/wn_network_top_{top_N}_{hrs:.0f}hr", cleanup=True)

    # Set up MCMC inference
    nuts_kernel = pyro.infer.NUTS(model, ignore_jit_warnings=True)
    mcmc = pyro.infer.MCMC(
        nuts_kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
    )
    mcmc.run(states, hrs, dt)
    mcmc.summary(prob=0.9)


if __name__ == "__main__":
    train_nominal()
