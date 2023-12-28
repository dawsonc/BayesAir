"""Run the simulation for a simple two-airport network."""
from itertools import combinations

import click
import matplotlib.pyplot as plt
import pandas as pd
import pyro
import seaborn as sns
import torch
import tqdm
from torch.nn.functional import sigmoid

import bayes_air.utils.dataloader as ba_dataloader
import wandb
from bayes_air.model import air_traffic_network_model
from bayes_air.network import NetworkState
from bayes_air.schedule import parse_schedule


def plot_travel_times(
    auto_guide, states, dt, n_samples, empirical_travel_times, wandb=True
):
    """Plot posterior samples of travel times."""
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
    subplot_spec = []
    for i in range(max_rows):
        subplot_spec.append(
            [f"{i * max_pairs_per_row +j}" for j in range(max_pairs_per_row)]
        )

    fig = plt.figure(layout="constrained", figsize=(12, 4 * max_rows))
    axs = fig.subplot_mosaic(subplot_spec)

    for i, pair in enumerate(pairs):
        # Put all of the data into a DataFrame to plot it
        plotting_df = pd.DataFrame(
            {
                f"{pair[0]}->{pair[1]}": 6.0
                * sigmoid(posterior_samples[f"travel_time_{pair[0]}_{pair[1]}"])
                .detach()
                .cpu()
                .numpy(),
                f"{pair[1]}->{pair[0]}": 6.0
                * sigmoid(posterior_samples[f"travel_time_{pair[1]}_{pair[0]}"])
                .detach()
                .cpu()
                .numpy(),
                "type": "Posterior",
            }
        )

        if not wandb:
            # Wandb doesn't support this KDE plot.
            sns.kdeplot(
                x=f"{pair[0]}->{pair[1]}",
                y=f"{pair[1]}->{pair[0]}",
                hue="type",
                ax=axs[f"{i}"],
                data=plotting_df,
                color="blue",
            )
            axs[f"{i}"].plot([], [], "-", color="blue", label="Posterior")
        else:
            axs[f"{i}"].scatter(
                plotting_df[f"{pair[0]}->{pair[1]}"],
                plotting_df[f"{pair[1]}->{pair[0]}"],
                marker=".",
                s=1,
                c="blue",
                label="Posterior",
                zorder=1,
            )

        axs[f"{i}"].scatter(
            empirical_travel_times.loc[
                (empirical_travel_times.origin_airport == pair[0])
                & (empirical_travel_times.destination_airport == pair[1]),
                "travel_time",
            ],
            empirical_travel_times.loc[
                (empirical_travel_times.origin_airport == pair[1])
                & (empirical_travel_times.destination_airport == pair[0]),
                "travel_time",
            ],
            marker="*",
            s=100,
            c="red",
            label="Empirical mean",
            zorder=10,
        )
        axs[f"{i}"].set_xlabel(f"{pair[0]} -> {pair[1]}")
        axs[f"{i}"].set_ylabel(f"{pair[1]} -> {pair[0]}")
        axs[f"{i}"].set_xlim(0, 8)
        axs[f"{i}"].set_ylim(0, 8)

        if i == 0:
            axs[f"{i}"].legend()
        else:
            axs[f"{i}"].legend([], [], frameon=False)

    return fig


def plot_service_times(auto_guide, states, dt, n_samples, wandb=True):
    """Plot posterior samples of service times."""
    # Sample mean service time estimates from the posterior
    with pyro.plate("samples", n_samples, dim=-1):
        posterior_samples = auto_guide(states, dt)

    # Make subplots for each airport
    airport_codes = states[0].airports.keys()
    n_pairs = len(airport_codes)
    max_rows = 3
    max_plots_per_row = n_pairs // max_rows + 1
    subplot_spec = []
    for i in range(max_rows):
        subplot_spec.append(
            [f"{i * max_plots_per_row +j}" for j in range(max_plots_per_row)]
        )

    fig = plt.figure(layout="constrained", figsize=(12, 4 * max_rows))
    axs = fig.subplot_mosaic(subplot_spec)

    for i, code in enumerate(airport_codes):
        # Put all of the data into a DataFrame to plot it
        plotting_df = pd.DataFrame(
            {
                code: 0.25
                * sigmoid(posterior_samples[f"{code}_mean_service_time"])
                .detach()
                .cpu()
                .numpy(),
                "type": "Posterior",
            }
        )

        sns.histplot(
            x=code,
            hue="type",
            ax=axs[f"{i}"],
            data=plotting_df,
            color="blue",
            kde=True,
        )
        axs[f"{i}"].set_xlim(-0.01, 0.26)

        if i == 0:
            axs[f"{i}"].legend()
        else:
            axs[f"{i}"].legend([], [], frameon=False)

    return fig


def train(top_n, days, svi_steps, n_samples, svi_lr, plot_every):
    pyro.clear_param_store()  # avoid leaking parameters across runs
    pyro.enable_validation(True)
    pyro.set_rng_seed(1)

    # Set the number of starting aircraft and crews at each airport
    starting_crew = 50
    starting_aircraft = 50

    # Hyperparameters
    dt = 0.2

    # Load the dataset
    df = pd.read_pickle("data/wn_data_clean.pkl")
    df = ba_dataloader.top_N_df(df, top_n)
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
    states = []  # Initialize wandb
    wandb.init(
        project="bayes-air",
        name=f"top-{top_n}-nominal-{days}-days",
        config={
            "type": "nominal",
            "starting_crew": starting_crew,
            "starting_aircraft": starting_aircraft,
            "dt": dt,
            "top_n": top_n,
            "days": days,
            "svi_lr": svi_lr,
            "svi_steps": svi_steps,
            "n_samples": n_samples,
        },
    )
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

    # Re-scale the ELBO by the number of days
    model = air_traffic_network_model
    model = pyro.poutine.scale(model, scale=1.0 / days)

    # Create an autoguide for the model
    auto_guide = pyro.infer.autoguide.AutoMultivariateNormal(model)

    # Set up SVI
    adam = pyro.optim.Adam({"lr": svi_lr})
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

    losses = []
    pbar = tqdm.tqdm(range(svi_steps))
    for i in pbar:
        loss = svi.step(states, dt)
        losses.append(loss)
        pbar.set_description(f"ELBO loss: {loss:.2f}")

        if i % plot_every == 0 or i == svi_steps - 1:
            fig = plot_travel_times(auto_guide, states, dt, n_samples, travel_times)
            wandb.log({"Travel times": fig}, commit=False)
            plt.close(fig)

            fig = plot_service_times(auto_guide, states, dt, n_samples, travel_times)
            wandb.log({"Service times": wandb.Image(fig)}, commit=False)
            plt.close(fig)

        wandb.log({"ELBO": loss})

    return loss


def train_from_config(config):
    return train(
        config.top_n,
        config.days,
        config.svi_steps,
        config.n_samples,
        config.svi_lr,
        config.plot_every,
    )


def train_sweep():
    wandb.init(project="bayes-air")
    train_from_config(wandb.config)


@click.command()
@click.option("--top-n", default=2, help="Number of airports to include in the network")
@click.option("--days", default=18, help="Number of days to simulate")
@click.option("--svi-steps", default=1000, help="Number of SVI steps to run")
@click.option("--n-samples", default=800, help="Number of posterior samples to draw")
@click.option("--svi-lr", default=1e-2, help="Learning rate for SVI")
@click.option("--plot-every", default=100, help="Plot every N steps")
def train_cmd(top_n, days, svi_steps, n_samples, svi_lr, plot_every):
    # Initialize wandb
    wandb.init(
        project="bayes-air",
        name=f"top-{top_n}-nominal-{days}-days",
        config={
            "type": "nominal",
            "top_n": top_n,
            "days": days,
            "svi_lr": svi_lr,
            "svi_steps": svi_steps,
            "n_samples": n_samples,
        },
    )

    train(top_n, days, svi_steps, n_samples, svi_lr, plot_every)


if __name__ == "__main__":
    # sweep_configuration = {
    #     "method": "grid",
    #     "metric": {"goal": "minimize", "name": "ELBO"},
    #     "parameters": {
    #         "top_n": {"values": [2]},
    #         "days": {"values": [2, 5, 10, 18]},
    #         "svi_steps": {"value": 10000},
    #         "n_samples": {"value": 800},
    #         "plot_every": {"value": 500},
    #         "svi_lr": {"values": [1e-2, 1e-3, 1e-4]},
    #     },
    # }

    # # Start the sweep
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="bayes-air")
    # wandb.agent(sweep_id, function=train_sweep, count=10)
    train_sweep()
