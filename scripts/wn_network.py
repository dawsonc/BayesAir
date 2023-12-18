"""Run the simulation for a simple two-airport network."""
import matplotlib.pyplot as plt
import pyro
import seaborn as sns
import torch
import tqdm

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
    hrs = 50.0  # extend a bit past 1 day to let all flights finish
    dt = 0.1
    top_N = 2
    days = 1
    svi_lr = 1e-2
    svi_steps = 500
    n_samples = 800

    # Load the dataset
    df = ba_dataloader.load_all_data()
    df = ba_dataloader.remap_columns(df)
    df = ba_dataloader.top_N_df(df, top_N)
    nominal_df, _ = ba_dataloader.split_nominal_disrupted_data(df)
    nominal_dfs = ba_dataloader.split_by_date(nominal_df)

    # Estimate travel times between each pair
    example_day_df = nominal_dfs[0]
    example_day_df["travel_time"] = (
        example_day_df["actual_arrival_time"] - example_day_df["actual_departure_time"]
    )
    travel_times = (
        example_day_df.groupby(["origin_airport", "destination_airport"])["travel_time"]
        .mean()
        .reset_index()
    )
    print("Travel times:")
    print(travel_times)

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

    states = states[:days]

    model = air_traffic_network_model

    # Create an autoguide for the model
    auto_guide = pyro.infer.autoguide.AutoMultivariateNormal(model)

    # Set up SVI
    adam = pyro.optim.Adam({"lr": svi_lr})
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

    losses = []
    pbar = tqdm.tqdm(range(svi_steps))
    for _ in pbar:
        loss = svi.step(states, hrs, dt)
        losses.append(loss)
        pbar.set_description(f"ELBO loss: {loss:.2f}")

    # Sample nominal travel time estimates from the posterior
    with pyro.plate("samples", n_samples, dim=-1):
        posterior_samples = auto_guide(states, hrs, dt)

    # Plot training curve and travel time posterior
    pairs = [
        ("DEN", "MDW"),
        # ("DEN", "LAS"),
        # ("LAS", "MDW"),
    ]

    _, axs = plt.subplots(1, 1 + len(pairs), figsize=(10, 4))

    axs[0].plot(losses)
    axs[0].set_xlabel("SVI step")
    axs[0].set_ylabel("ELBO loss")

    for i, pair in enumerate(pairs):
        sns.kdeplot(
            x=posterior_samples[f"travel_time_{pair[0]}_{pair[1]}"]
            .detach()
            .cpu()
            .numpy(),
            y=posterior_samples[f"travel_time_{pair[1]}_{pair[0]}"]
            .detach()
            .cpu()
            .numpy(),
            ax=axs[i + 1],
        )
        axs[i + 1].set_xlabel(f"Travel time {pair[0]} -> {pair[1]}")
        axs[i + 1].set_ylabel(f"Travel time {pair[1]} -> {pair[0]}")
        axs[i + 1].set_xlim(0, 5)
        axs[i + 1].set_ylim(0, 5)

    plt.show()


if __name__ == "__main__":
    train_nominal()
