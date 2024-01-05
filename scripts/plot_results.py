import os
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import pyro
import seaborn as sns
import torch

import bayes_air.utils.dataloader as ba_dataloader
from bayes_air.model import air_traffic_network_model
from bayes_air.network import NetworkState
from bayes_air.schedule import parse_schedule


def posterior_samples_from_checkpoint(
    states, dt, checkpoint_dir, num_samples=1000, epoch=1999
):
    pyro.clear_param_store()  # avoid leaking parameters across runs
    pyro.enable_validation(True)
    pyro.set_rng_seed(1)

    # Load the params
    pyro.get_param_store().load(os.path.join(checkpoint_dir, str(epoch), "params.pth"))

    # Create the model and autoguide objects
    model = air_traffic_network_model
    model = pyro.poutine.scale(model, scale=1.0 / len(states))
    auto_guide = pyro.infer.autoguide.AutoMultivariateNormal(model)

    # # Load the checkpoint
    # auto_guide.load_state_dict(
    #     torch.load(
    #         os.path.join(checkpoint_dir, str(epoch), "guide.pth"), map_location="cpu"
    #     )
    # )

    # Sample from the posterior
    with pyro.plate("samples", num_samples, dim=-1):
        posterior_samples = auto_guide(states, dt)

    return posterior_samples


def generate_posterior_samples_dataframe(
    nominal_dfs,
    disrupted_dfs,
    dt,
    start_days,
    duration,
    checkpoint_base_dir,
    top_N,
    num_samples,
):
    """
    Generate posterior samples from each day of the nominal and disrupted data.
    """
    results_df = pd.DataFrame()

    # Add the nominal data
    for day in start_days:
        checkpoint_dir = os.path.join(
            checkpoint_base_dir, f"top_{top_N}_nominal_days_{day}_{day + duration}"
        )

        states = []
        for day_df in nominal_dfs[day : day + duration]:
            flights, airports = parse_schedule(day_df)

            state = NetworkState(
                airports={airport.code: airport for airport in airports},
                pending_flights=flights,
            )
            states.append(state)

        posterior_samples = posterior_samples_from_checkpoint(
            states, dt, checkpoint_dir, num_samples
        )

        # Make into a dataframe
        airport_codes = states[0].airports.keys()
        pairs = list(combinations(airport_codes, 2))
        posterior_df = pd.DataFrame(
            {
                f"{code} mean service time (hr)": posterior_samples[
                    f"{code}_mean_service_time"
                ]
                .detach()
                .cpu()
                .numpy()
                for code in airport_codes
            }
            | {
                f"{code} initial aircraft reserve": torch.exp(
                    posterior_samples[f"{code}_log_initial_available_aircraft"]
                )
                .detach()
                .cpu()
                .numpy()
                for code in airport_codes
            }
            | {
                f"{pair[0]}->{pair[1]} travel time (hr)": posterior_samples[
                    f"travel_time_{pair[0]}_{pair[1]}"
                ]
                .detach()
                .cpu()
                .numpy()
                for pair in pairs
            }
            | {
                f"{pair[1]}->{pair[0]} travel time (hr)": posterior_samples[
                    f"travel_time_{pair[1]}_{pair[0]}"
                ]
                .detach()
                .cpu()
                .numpy()
                for pair in pairs
            }
            | {
                "State": "Nominal",
                "Day": day,
                "Duration": duration,
                "Date": nominal_dfs[day].date.iloc[0],
            }
        )
        results_df = pd.concat([results_df, posterior_df], ignore_index=True)

    # Same for the disrupted data
    for day in start_days:
        checkpoint_dir = os.path.join(
            checkpoint_base_dir, f"top_{top_N}_disrupted_days_{day}_{day + duration}"
        )

        states = []
        for day_df in disrupted_dfs[day : day + duration]:
            flights, airports = parse_schedule(day_df)

            state = NetworkState(
                airports={airport.code: airport for airport in airports},
                pending_flights=flights,
            )
            states.append(state)

        posterior_samples = posterior_samples_from_checkpoint(
            states, dt, checkpoint_dir, num_samples
        )

        # Make into a dataframe
        airport_codes = states[0].airports.keys()
        pairs = list(combinations(airport_codes, 2))
        posterior_df = pd.DataFrame(
            {
                f"{code} mean service time (hr)": posterior_samples[
                    f"{code}_mean_service_time"
                ]
                .detach()
                .cpu()
                .numpy()
                for code in airport_codes
            }
            | {
                f"{code} initial aircraft reserve": torch.exp(
                    posterior_samples[f"{code}_log_initial_available_aircraft"]
                )
                .detach()
                .cpu()
                .numpy()
                for code in airport_codes
            }
            | {
                f"{pair[0]}->{pair[1]} travel time (hr)": posterior_samples[
                    f"travel_time_{pair[0]}_{pair[1]}"
                ]
                .detach()
                .cpu()
                .numpy()
                for pair in pairs
            }
            | {
                f"{pair[1]}->{pair[0]} travel time (hr)": posterior_samples[
                    f"travel_time_{pair[1]}_{pair[0]}"
                ]
                .detach()
                .cpu()
                .numpy()
                for pair in pairs
            }
            | {
                "State": "Disrupted",
                "Day": day,
                "Duration": duration,
                "Date": disrupted_dfs[day].date.iloc[0],
            }
        )
        results_df = pd.concat([results_df, posterior_df], ignore_index=True)

    return results_df


def save_posterior_df_to_disk(base_path, num_samples):
    """
    Generate posterior samples from each day of the nominal and disrupted data.
    """
    # Hyperparameters
    dt = 0.2
    start_days = range(7)
    duration = 1
    top_N = 3
    checkpoint_base_dir = "checkpoints"

    # Load the dataset
    df = pd.read_pickle("data/wn_data_clean.pkl")
    df = ba_dataloader.top_N_df(df, top_N)
    nominal_df, disrupted_df = ba_dataloader.split_nominal_disrupted_data(df)
    nominal_dfs = ba_dataloader.split_by_date(nominal_df)
    disrupted_dfs = ba_dataloader.split_by_date(disrupted_df)

    results_df = generate_posterior_samples_dataframe(
        nominal_dfs,
        disrupted_dfs,
        dt,
        start_days,
        duration,
        checkpoint_base_dir,
        top_N,
        num_samples,
    )

    # Save to disk
    results_df.to_csv(
        os.path.join(base_path, f"top_{top_N}_posterior_samples.csv"),
        index=False,
    )


def load_posterior_df(base_path):
    """
    Load the posterior samples dataframe from disk.
    """
    top_N = 3
    return pd.read_csv(os.path.join(base_path, f"top_{top_N}_posterior_samples.csv"))


if __name__ == "__main__":
    generate = False
    if generate:
        save_posterior_df_to_disk("analysis", num_samples=500)

    df = load_posterior_df("analysis")

    # Convert to date
    df["Date"] = pd.to_datetime(df["Date"])
    df["Day of week"] = df["Date"].dt.day_name()

    sns.scatterplot(
        data=df,
        x="LAS->MDW travel time (hr)",
        y="MDW->LAS travel time (hr)",
        hue="State",
        s=2,
    )
    plt.savefig("analysis/travel_times.png")
    plt.close()

    airports = ["DEN", "LAS", "MDW"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    for i, airport in enumerate(airports):
        sns.violinplot(
            data=df,
            x="Day of week",
            y=f"{airport} mean service time (hr)",
            hue="State",
            # alpha=0.2,
            # legend=False,
            order=[
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
                "Monday",
                "Tuesday",
            ],
            split=True,
            gap=0.1,
            inner="quart",
            cut=0,
            ax=axs[i],
        )
        plt.xlabel("")
        if i < len(airports) - 1:
            axs[i].set_xticklabels([])

    plt.savefig("analysis/service_times.png")
    plt.close()

    airports = ["DEN", "LAS", "MDW"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    for i, airport in enumerate(airports):
        sns.violinplot(
            data=df,
            x="Day of week",
            y=f"{airport} initial aircraft reserve",
            hue="State",
            # alpha=0.2,
            legend=i == 0,
            order=[
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
                "Monday",
                "Tuesday",
            ],
            split=True,
            gap=0.1,
            inner="quart",
            cut=0,
            ax=axs[i],
        )
        plt.xlabel("")
        if i < len(airports) - 1:
            axs[i].set_xticklabels([])

    plt.savefig("analysis/reserves.png")
    plt.close()
