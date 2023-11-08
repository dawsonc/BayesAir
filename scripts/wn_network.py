"""Run the simulation for a simple two-airport network."""
import pandas as pd
import pyro
import torch
from tqdm import tqdm

from bayes_air.model import air_traffic_network_model
from bayes_air.network import NetworkState
from bayes_air.schedule import parse_schedule
import bayes_air.utils.dataloader as ba_dataloader


def train_nominal():
    pyro.enable_validation(True)
    pyro.set_rng_seed(1)

    # Set the number of starting aircraft and crews at each airport
    starting_crew = 50
    starting_aircraft = 50

    # Load the dataset
    df = ba_dataloader.load_all_data()
    nominal_df, _ = ba_dataloader.split_nominal_disrupted_data(df)
    nominal_dfs = ba_dataloader.split_by_date(nominal_df)
    nominal_dfs = [ba_dataloader.remap_columns(df) for df in nominal_dfs]

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
    guide = pyro.infer.autoguide.AutoMultivariateNormal(model)

    # Render the first few hours of the model to see what the structure is
    hrs = 6.0
    model_graph = pyro.render_model(
        air_traffic_network_model,
        model_args=(states, hrs, 0.1),
        render_params=False,
        render_distributions=False,
    )
    model_graph.render(f"tmp/wn_network_{hrs:.0f}hr")

    # Render the guide too
    guide_graph = pyro.render_model(
        guide,
        model_args=(states, hrs, 0.1),
        render_params=False,
        render_distributions=False,
    )
    guide_graph.render(f"tmp/wn_network_{hrs:.0f}hr_guide")

    # setup the optimizer
    adam_params = {"lr": 0.0001}
    optimizer = pyro.optim.Adam(adam_params)

    # setup the inference algorithm
    svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())

    # do gradient steps
    n_steps = 1000
    losses = []
    pbar = tqdm(range(n_steps))
    for step in pbar:
        loss = svi.step(states[:1], hrs, 0.1)  # Only do the first day to start
        losses.append(loss)
        pbar.set_description(f"ELBO Loss: {loss:.2f}")


if __name__ == "__main__":
    train_nominal()
