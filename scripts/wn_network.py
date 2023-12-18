"""Run the simulation for a simple two-airport network."""
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
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
    dt = 0.2
    top_N = 6
    days = 1
    svi_lr = 3e-3
    svi_steps = 1000
    n_samples = 800

    # Load the dataset
    df = pd.read_pickle("data/wn_data_clean.csv")
    df = ba_dataloader.top_N_df(df, top_N)
    nominal_df, _ = ba_dataloader.split_nominal_disrupted_data(df)
    nominal_dfs = ba_dataloader.split_by_date(nominal_df)

    num_flights = sum([len(df) for df in nominal_dfs[:days]])
    print(f"Number of flights: {num_flights}")

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

    # Convert each day into a schedule
    states = []
    for day_df in nominal_dfs[:days]:
        flights, airports = parse_schedule(day_df)

        # Add starting aircraft and crew to each airport
        for airport in airports:
            for _ in range(starting_aircraft):
                airport.available_aircraft.append(torch.tensor(0.0))
            for _ in range(starting_crew):
                airport.available_crew.append(torch.tensor(0.0))

        state = NetworkState(
            airports={airport.code: airport for airport in airports},
            pending_flights=flights,
        )
        states.append(state)

    model = air_traffic_network_model

    # Create an autoguide for the model
    auto_guide = pyro.infer.autoguide.AutoMultivariateNormal(
        model, init_loc_fn=pyro.infer.autoguide.init_to_mean
    )

    # Set up SVI
    adam = pyro.optim.Adam({"lr": svi_lr})
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

    losses = []
    pbar = tqdm.tqdm(range(svi_steps))
    for _ in pbar:
        loss = svi.step(states, dt)
        losses.append(loss)
        pbar.set_description(f"ELBO loss: {loss:.2f}")

    # Sample nominal travel time estimates from the posterior
    with pyro.plate("samples", n_samples, dim=-1):
        posterior_samples = auto_guide(states, dt)

    # Plot training curve and travel time posterior
    airport_codes = states[0].airports.keys()
    pairs = list(combinations(airport_codes, 2))

    # Make subplots for the learning curve and each travel time pair
    n_pairs = len(pairs)
    max_rows = 3
    max_pairs_per_row = n_pairs // max_rows + 1
    subplot_spec = [
        ["loss", "loss"] + ["." for _ in range(max(max_pairs_per_row - 2, 0))]
    ]
    for i in range(max_rows):
        subplot_spec.append(
            [f"{i * max_pairs_per_row +j}" for j in range(max_pairs_per_row)]
        )

    fig = plt.figure(layout="constrained")
    axs = fig.subplot_mosaic(subplot_spec)

    axs["loss"].plot(losses)
    axs["loss"].set_xlabel("SVI step")
    axs["loss"].set_ylabel("ELBO loss")

    for i, pair in enumerate(pairs):
        # Put all of the data into a DataFrame to plot it
        plotting_df = pd.DataFrame(
            {
                f"{pair[0]}->{pair[1]}": posterior_samples[
                    f"travel_time_{pair[0]}_{pair[1]}"
                ]
                .detach()
                .cpu()
                .numpy(),
                f"{pair[1]}->{pair[0]}": posterior_samples[
                    f"travel_time_{pair[1]}_{pair[0]}"
                ]
                .detach()
                .cpu()
                .numpy(),
                "type": "Posterior",
            }
        )

        sns.kdeplot(
            x=f"{pair[0]}->{pair[1]}",
            y=f"{pair[1]}->{pair[0]}",
            hue="type",
            ax=axs[f"{i}"],
            data=plotting_df,
            color="blue",
        )
        axs[f"{i}"].scatter(
            travel_times.loc[
                (travel_times.origin_airport == pair[0])
                & (travel_times.destination_airport == pair[1]),
                "travel_time",
            ],
            travel_times.loc[
                (travel_times.origin_airport == pair[1])
                & (travel_times.destination_airport == pair[0]),
                "travel_time",
            ],
            marker="*",
            s=100,
            c="red",
            label="Empirical mean",
            zorder=10,
        )
        axs[f"{i}"].plot([], [], "-", c=sns.color_palette()[0], label="Posterior")
        axs[f"{i}"].set_xlabel(f"{pair[0]} -> {pair[1]}")
        axs[f"{i}"].set_ylabel(f"{pair[1]} -> {pair[0]}")
        axs[f"{i}"].set_xlim(0, 11)
        axs[f"{i}"].set_ylim(0, 11)

        if i == 0:
            axs[f"{i}"].legend()
        else:
            axs[f"{i}"].legend([], [], frameon=False)

    plt.show()


if __name__ == "__main__":
    train_nominal()
