"""Run the simulation for a simple two-airport network."""
import os
from itertools import combinations

import click
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pyro
import seaborn as sns
import torch
import tqdm

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


def plot_starting_aircraft(auto_guide, states, dt, n_samples):
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
                code: torch.exp(
                    posterior_samples[f"{code}_log_initial_available_aircraft"]
                )
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
            # kde=True,
        )
        axs[f"{i}"].set_xlim(-0.05, 30.0)

    return fig


def plot_service_times(auto_guide, states, dt, n_samples):
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
                code: posterior_samples[f"{code}_mean_service_time"]
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
            # kde=True,
        )
        axs[f"{i}"].set_xlim(-0.05, 1.05)

    return fig


def train(
    top_n, days, svi_steps, n_samples, svi_lr, plot_every, start_day=0, nominal=True
):
    pyro.clear_param_store()  # avoid leaking parameters across runs
    pyro.enable_validation(True)
    pyro.set_rng_seed(1)

    # Avoid plotting error
    matplotlib.use("Agg")

    # Set the number of starting aircraft at each airport
    starting_aircraft = 50

    # Hyperparameters
    dt = 0.2

    # Load the dataset
    df = pd.read_pickle("data/wn_data_clean.pkl")
    df = ba_dataloader.top_N_df(df, top_n)
    nominal_df, disrupted_df = ba_dataloader.split_nominal_disrupted_data(df)
    nominal_dfs = ba_dataloader.split_by_date(nominal_df)
    disrupted_dfs = ba_dataloader.split_by_date(disrupted_df)

    # Use nominal or disrupted data
    if nominal:
        data = nominal_dfs
    else:
        data = disrupted_dfs

    num_flights = sum([len(df) for df in data[start_day : start_day + days]])
    print(f"Number of flights: {num_flights}")

    # Estimate travel times between each pair
    example_day_df = data[start_day]
    example_day_df["travel_time"] = (
        example_day_df["actual_arrival_time"] - example_day_df["actual_departure_time"]
    )
    travel_times = (
        example_day_df.groupby(["origin_airport", "destination_airport"])["travel_time"]
        .mean()
        .reset_index()
    )

    # Initialize wandb
    run_name = f"top_{top_n}_"
    run_name += f"{'nominal' if nominal else 'disrupted'}_"
    run_name += f"days_{start_day}_{start_day + days}"
    wandb.init(
        project="bayes-air",
        name=run_name,
        group="nominal" if nominal else "disrupted",
        config={
            "type": "nominal",
            "starting_aircraft": starting_aircraft,
            "dt": dt,
            "top_n": top_n,
            "days": days,
            "svi_lr": svi_lr,
            "svi_steps": svi_steps,
            "n_samples": n_samples,
        },
    )

    # Convert each day into a schedule
    states = []
    for day_df in data[start_day : start_day + days]:
        flights, airports = parse_schedule(day_df)

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
    # auto_guide = pyro.infer.autoguide.AutoDelta(model)

    # Set up SVI
    gamma = 0.1  # final learning rate will be gamma * initial_lr
    lrd = gamma ** (1 / svi_steps)
    optim = pyro.optim.ClippedAdam({"lr": svi_lr, "lrd": lrd})
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, auto_guide, optim, elbo)

    losses = []
    pbar = tqdm.tqdm(range(svi_steps))
    for i in pbar:
        loss = svi.step(states, dt)
        losses.append(loss)
        pbar.set_description(f"ELBO loss: {loss:.2f}")

        if i % plot_every == 0 or i == svi_steps - 1:
            fig = plot_travel_times(auto_guide, states, dt, n_samples, travel_times)
            wandb.log({"Travel times": wandb.Image(fig)}, commit=False)
            plt.close(fig)

            fig = plot_service_times(auto_guide, states, dt, n_samples)
            wandb.log({"Mean service times": wandb.Image(fig)}, commit=False)
            plt.close(fig)

            fig = plot_starting_aircraft(auto_guide, states, dt, n_samples)
            wandb.log({"Starting aircraft": wandb.Image(fig)}, commit=False)
            plt.close(fig)

            # Save the params and autoguide
            dir_path = os.path.dirname(__file__)
            save_path = os.path.join(dir_path, "..", "checkpoints", run_name, f"{i}")
            os.makedirs(save_path, exist_ok=True)
            pyro.get_param_store().save(os.path.join(save_path, "params.pth"))
            torch.save(auto_guide.state_dict(), os.path.join(save_path, "guide.pth"))

        wandb.log({"ELBO": loss})

    return loss


@click.command()
@click.option("--disrupted", is_flag=True, help="Use disrupted data")
@click.option("--top-n", default=2, help="Number of airports to include in the network")
@click.option("--days", default=2, help="Number of days to simulate")
@click.option("--start-day", default=0, help="Index of day to start with")
@click.option("--svi-steps", default=1000, help="Number of SVI steps to run")
@click.option("--n-samples", default=800, help="Number of posterior samples to draw")
@click.option("--svi-lr", default=1e-3, help="Learning rate for SVI")
@click.option("--plot-every", default=100, help="Plot every N steps")
def train_cmd(
    disrupted, top_n, days, start_day, svi_steps, n_samples, svi_lr, plot_every
):
    train(
        top_n,
        days,
        svi_steps,
        n_samples,
        svi_lr,
        plot_every,
        start_day=start_day,
        nominal=not disrupted,
    )


if __name__ == "__main__":
    train_cmd()
