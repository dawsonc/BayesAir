"""Implement CalVI training for the southwest network problem."""
import os
from itertools import combinations
from math import ceil

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pyro
import torch
import zuko
from click import command, option

import bayes_air.utils.dataloader as ba_dataloader
import wandb
from bayes_air.model import air_traffic_network_model
from bayes_air.network import NetworkState
from bayes_air.schedule import parse_schedule
from scripts.training import train
from scripts.utils import kl_divergence


@command()
@option("--top-n", default=4, help="# of airports to include")
@option(
    "--include-cancellations",
    is_flag=True,
    help="If true, include a crew model and cancellations",
)
@option("--n-nominal", default=18, help="# of nominal examples")
@option("--n-failure", default=2, help="# of failure examples for training")
@option("--n-failure-eval", default=2, help="# of failure examples for evaluation")
@option("--no-calibrate", is_flag=True, help="Don't use calibration")
@option("--regularize", is_flag=True, help="Regularize failure using KL wrt nominal")
@option("--wasserstein", is_flag=True, help="Regularize failure using W2 wrt nominal")
@option("--seed", default=0, help="Random seed")
@option("--n-steps", default=400, type=int, help="# of steps")
@option("--lr", default=1e-3, type=float, help="Learning rate")
@option("--lr-gamma", default=1.0, type=float, help="Learning rate decay")
@option("--lr-steps", default=1000, type=int, help="Steps per learning rate decay")
@option("--grad-clip", default=10, type=float, help="Gradient clipping value")
@option("--weight-decay", default=0.0, type=float, help="Weight decay rate")
@option("--run-prefix", default="", help="Prefix for run name")
@option(
    "--n-elbo-particles",
    default=1,
    type=int,
    help="# of particles for ELBO estimation",
)
@option(
    "--n-calibration-particles",
    default=50,
    type=int,
    help="# of particles for calibration",
)
@option(
    "--n-calibration-permutations",
    default=5,
    type=int,
    help="# of permutations for calibration",
)
@option(
    "--n-divergence-particles",
    default=100,
    type=int,
    help="# of particles for divergence estimation",
)
@option(
    "--calibration-weight",
    default=1e0,
    type=float,
    help="weight applied to calibration loss",
)
@option(
    "--regularization-weight",
    default=1e0,
    type=float,
    help="weight applied to nominal divergence loss",
)
@option("--elbo-weight", default=1e0, type=float, help="weight applied to ELBO loss")
@option(
    "--calibration-ub", default=5e1, type=float, help="KL upper bound for calibration"
)
@option("--calibration-lr", default=1e-3, type=float, help="LR for calibration")
@option("--calibration-substeps", default=1, type=int, help="# of calibration substeps")
@option(
    "--calibration-steps",
    default=5,
    type=int,
    help="# of calibration steps for evaluation",
)
def run(
    top_n,
    include_cancellations,
    n_nominal,
    n_failure,
    n_failure_eval,
    no_calibrate,
    regularize,
    wasserstein,
    seed,
    n_steps,
    lr,
    lr_gamma,
    lr_steps,
    grad_clip,
    weight_decay,
    run_prefix,
    n_elbo_particles,
    n_calibration_particles,
    n_calibration_permutations,
    n_divergence_particles,
    calibration_weight,
    regularization_weight,
    elbo_weight,
    calibration_ub,
    calibration_lr,
    calibration_substeps,
    calibration_steps,
):
    """Generate data and train the SWI model."""
    matplotlib.use("Agg")
    matplotlib.rcParams["figure.dpi"] = 300

    # Parse arguments
    calibrate = not no_calibrate

    # Generate data (use the same seed for all runs)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # WN model not GPU-ized yet
    torch.manual_seed(0)
    pyro.set_rng_seed(0)

    # Load the data
    df = pd.read_pickle("data/wn_data_clean.pkl")
    df = ba_dataloader.top_N_df(df, top_n)
    nominal_df, disrupted_df = ba_dataloader.split_nominal_disrupted_data(df)
    nominal_dfs = ba_dataloader.split_by_date(nominal_df)
    disrupted_dfs = ba_dataloader.split_by_date(disrupted_df)

    # Get just the set of data we want to study
    nominal = nominal_dfs[-n_nominal:]
    failure = disrupted_dfs[: 2 * n_failure : 2]
    n_failure_eval = n_failure
    failure_eval = disrupted_dfs[1 : 1 + n_failure : 2]

    # Filter out cancellations if we're not using them
    if not include_cancellations:
        nominal = [df[~df["cancelled"]] for df in nominal]
        failure = [df[~df["cancelled"]] for df in failure]
        failure_eval = [df[~df["cancelled"]] for df in failure_eval]

    # Convert each day into a schedule
    nominal_states = []
    failure_states = []
    failure_eval_states = []

    for day_df in nominal:
        flights, airports = parse_schedule(day_df)

        state = NetworkState(
            airports={airport.code: airport for airport in airports},
            pending_flights=flights,
        )
        nominal_states.append(state)

    for day_df in failure:
        flights, airports = parse_schedule(day_df)

        state = NetworkState(
            airports={airport.code: airport for airport in airports},
            pending_flights=flights,
        )
        failure_states.append(state)

    for day_df in failure_eval:
        flights, airports = parse_schedule(day_df)

        state = NetworkState(
            airports={airport.code: airport for airport in airports},
            pending_flights=flights,
        )
        failure_eval_states.append(state)

    # Get some information about the network that will be needed to map
    # the vector posterior to the sample sites in the probabilistic model
    airport_codes = list(nominal_states[0].airports.keys())
    n_airports = len(airport_codes)
    pairs = list(combinations(airport_codes, 2))
    n_latent_variables = (
        n_airports  # mean turnaround time for each airport
        + n_airports  # mean service time for each airport
        + n_airports * n_airports  # travel time between each pair of airports
    )
    if include_cancellations:
        n_latent_variables += n_airports  # log # of initial aircraft for each airport

    # Fixed model parameter: timestep
    dt = 0.2

    # Vary the seed for training
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    def map_to_sample_sites(sample):
        """Map vectorized samples to sample sites in the probabilistic model."""
        # Handle batched samples
        single_sample = len(sample.shape) == 1
        if single_sample:
            sample = sample.unsqueeze(0)

        assert sample.shape[-1] == n_latent_variables

        # Reshape & reparameterize the sample to satisfy positivity constraints
        airport_turnaround_times = torch.exp(sample[:, :n_airports])
        airport_service_times = torch.exp(sample[:, n_airports : 2 * n_airports])
        if include_cancellations:
            log_airport_initial_available_aircraft = sample[
                :, 2 * n_airports : 3 * n_airports
            ]
            travel_times = torch.exp(
                sample[:, 3 * n_airports :].reshape(-1, n_airports, n_airports)
            )
        else:
            travel_times = torch.exp(
                sample[:, 2 * n_airports :].reshape(-1, n_airports, n_airports)
            )

        # Map to sample sites in the model
        conditioning_dict = {}
        for i, code in enumerate(airport_codes):
            conditioning_dict[
                f"{code}_mean_turnaround_time"
            ] = airport_turnaround_times[:, i]
            conditioning_dict[f"{code}_mean_service_time"] = airport_service_times[:, i]
            if include_cancellations:
                conditioning_dict[
                    f"{code}_log_initial_available_aircraft"
                ] = log_airport_initial_available_aircraft[:, i]

        for i, origin in enumerate(airport_codes):
            for j, destination in enumerate(airport_codes):
                if origin != destination:
                    conditioning_dict[
                        f"travel_time_{origin}_{destination}"
                    ] = travel_times[:, i, j]

        # Remove the batch dimension if it wasn't there before
        if single_sample:
            conditioning_dict = {
                key: value.squeeze(0) for key, value in conditioning_dict.items()
            }

        return conditioning_dict

    # Make the objective and divergence closures
    def objective_fn(guide_dist, _, states):
        """ELBO loss for the air traffic problem."""
        elbo = torch.tensor(0.0).to(device)
        for _ in range(n_elbo_particles):
            posterior_sample, posterior_logprob = guide_dist.rsample_and_log_prob()

            conditioning_dict = map_to_sample_sites(posterior_sample)

            model_trace = pyro.poutine.trace(
                pyro.poutine.condition(
                    air_traffic_network_model, data=conditioning_dict
                )
            ).get_trace(
                states=states,
                delta_t=dt,
                device=device,
                include_cancellations=include_cancellations,
            )
            model_logprob = model_trace.log_prob_sum()

            elbo += (model_logprob - posterior_logprob) / n_elbo_particles

        # Make it negative to make it a loss and scale by the number of flights
        num_flights = sum(len(state.pending_flights) for state in states)
        return -elbo / num_flights

    def divergence_fn(p, q):
        """Compute the KL divergence"""
        return kl_divergence(p, q, n_divergence_particles)

    # Define plotting callbacks
    @torch.no_grad()
    def plot_travel_times(*sample_maps, labels=None):
        # Make subplots for each travel time pair
        n_pairs = len(pairs)
        max_rows = 2
        max_pairs_per_row = n_pairs // max_rows + 1
        subplot_spec = []
        for i in range(max_rows):
            subplot_spec.append(
                [f"{i * max_pairs_per_row +j}" for j in range(max_pairs_per_row)]
            )

        fig = plt.figure(layout="constrained", figsize=(12, 4 * max_rows))
        axs = fig.subplot_mosaic(subplot_spec)

        for i, pair in enumerate(pairs):
            for j, sample_map in enumerate(sample_maps):
                axs[f"{i}"].scatter(
                    sample_map[f"travel_time_{pair[0]}_{pair[1]}"].cpu(),
                    sample_map[f"travel_time_{pair[1]}_{pair[0]}"].cpu(),
                    marker=".",
                    s=1,
                    label=labels[j] if labels else None,
                )

            axs[f"{i}"].set_xlabel(f"{pair[0]} -> {pair[1]} travel time (hr)")
            axs[f"{i}"].set_ylabel(f"{pair[1]} -> {pair[0]} travel time (hr)")
            axs[f"{i}"].set_xlim(0, 8)
            axs[f"{i}"].set_ylim(0, 8)
            axs[f"{i}"].legend()

        return fig

    @torch.no_grad()
    def plot_initial_aircraft(*sample_maps, labels=None):
        # Make subplots for each airport
        max_rows = 2
        max_plots_per_row = ceil(n_airports / max_rows)
        subplot_spec = []
        for i in range(max_rows):
            subplot_spec.append(
                [f"{i * max_plots_per_row +j}" for j in range(max_plots_per_row)]
            )

        fig = plt.figure(layout="constrained", figsize=(12, 4 * max_rows))
        axs = fig.subplot_mosaic(subplot_spec)

        for i, code in enumerate(airport_codes):
            for j, sample_map in enumerate(sample_maps):
                axs[f"{i}"].hist(
                    torch.exp(
                        sample_map[f"{code}_log_initial_available_aircraft"]
                    ).cpu(),
                    bins=64,
                    label=labels[j] if labels else None,
                    alpha=1 / len(sample_maps),
                )

            axs[f"{i}"].set_xlabel(f"{code} aircraft reserve")
            axs[f"{i}"].set_xlim(-0.05, 30.0)
            axs[f"{i}"].legend()

        return fig

    @torch.no_grad()
    def plot_service_times(*sample_maps, labels=None):
        # Make subplots for each airport
        max_rows = 2
        max_plots_per_row = ceil(n_airports / max_rows)
        subplot_spec = []
        for i in range(max_rows):
            subplot_spec.append(
                [f"{i * max_plots_per_row +j}" for j in range(max_plots_per_row)]
            )

        fig = plt.figure(layout="constrained", figsize=(12, 4 * max_rows))
        axs = fig.subplot_mosaic(subplot_spec)

        for i, code in enumerate(airport_codes):
            for j, sample_map in enumerate(sample_maps):
                axs[f"{i}"].hist(
                    sample_map[f"{code}_mean_service_time"].cpu(),
                    bins=64,
                    label=labels[j] if labels else None,
                    alpha=1 / len(sample_maps),
                )

            axs[f"{i}"].set_xlabel(f"{code} service time (hr)")
            x_min, x_max = axs[f"{i}"].get_xlim()
            x_min = min(x_min, -0.05)
            x_max = max(x_max, 1.05)
            axs[f"{i}"].set_xlim(x_min, x_max)
            axs[f"{i}"].legend()

        return fig

    @torch.no_grad()
    def plot_posterior(*dists, labels=None, save_file_name=None, save_wandb=False):
        """Make a couple of plots for the posterior. Does not save to disk."""
        # Generate a bunch of samples
        samples = [dist.sample((1000,)) for dist in dists]
        sample_maps = [map_to_sample_sites(sample) for sample in samples]

        # Make some plots
        fig = plot_travel_times(*sample_maps, labels=labels)
        if save_wandb:
            wandb.log({"Posterior travel times": wandb.Image(fig)}, commit=False)
        plt.close(fig)

        if include_cancellations:
            fig = plot_initial_aircraft(*sample_maps, labels=labels)
            if save_wandb:
                wandb.log(
                    {"Posterior starting aircraft": wandb.Image(fig)}, commit=False
                )
            plt.close(fig)

        fig = plot_service_times(*sample_maps, labels=labels)
        if save_wandb:
            wandb.log({"Posterior service times": wandb.Image(fig)}, commit=False)
        plt.close(fig)

    @torch.no_grad()
    def plot_posterior_grid(
        failure_guide,
        nominal_label,
        save_file_name=None,
        save_wandb=False,
    ):
        """Plot DEN service times on a grid"""
        n_steps = 5
        fig, axs = plt.subplots(
            n_steps,
            n_calibration_permutations,
            figsize=(5 * n_calibration_permutations, 5 * n_steps),
        )

        for row, j in enumerate(torch.linspace(0, 1, n_steps)):
            for i in range(n_calibration_permutations):
                label = torch.zeros(n_calibration_permutations).to(nominal_label.device)
                label[i] = j

                samples = failure_guide(label).sample((1000,))
                sample_map = map_to_sample_sites(samples)

                code = "DEN"
                axs[row, i].hist(sample_map[f"{code}_mean_service_time"].cpu(), bins=64)
                axs[row, i].set_xlabel(f"{code} service time (hr)")
                axs[row, i].set_xlim(-0.05, 1.05)

        if save_file_name:
            plt.savefig(save_file_name, bbox_inches="tight", dpi=300)

        if save_wandb:
            wandb.log({"Posterior grid": wandb.Image(fig)}, commit=False)

        plt.close()

    # Start wandb
    run_name = run_prefix
    run_name += "ours_" if (calibrate and not regularize) else ""
    run_name += "calibrated_" if calibrate else ""
    if regularize:
        run_name += "kl_regularized_kl" if not wasserstein else "w2_regularized"
    wandb.init(
        project="wn-1",
        name=run_name,
        group=run_name,
        config={
            "top_n": top_n,
            "include_cancellations": include_cancellations,
            "n_nominal": n_nominal,
            "n_failure": n_failure,
            "n_failure_eval": n_failure_eval,
            "no_calibrate": no_calibrate,
            "regularize": regularize,
            "wasserstein": wasserstein,
            "seed": seed,
            "n_steps": n_steps,
            "lr": lr,
            "lr_gamma": lr_gamma,
            "lr_steps": lr_steps,
            "grad_clip": grad_clip,
            "weight_decay": weight_decay,
            "n_elbo_particles": n_elbo_particles,
            "n_calibration_particles": n_calibration_particles,
            "n_calibration_permutations": n_calibration_permutations,
            "n_divergence_particles": n_divergence_particles,
            "calibration_weight": calibration_weight,
            "regularization_weight": regularization_weight,
            "elbo_weight": elbo_weight,
            "calibration_ub": calibration_ub,
            "calibration_lr": calibration_lr,
            "calibration_substeps": calibration_substeps,
            "calibration_steps": calibration_steps,
        },
    )

    # Make a directory for checkpoints if it doesn't already exist
    os.makedirs(f"checkpoints/swi/{run_name}", exist_ok=True)

    # Initialize the models
    if wasserstein:
        failure_guide = zuko.flows.CNF(
            features=n_latent_variables,
            context=n_calibration_permutations,
            hidden_features=(64, 64),
        ).to(device)
    else:
        failure_guide = zuko.flows.NSF(
            features=n_latent_variables,
            context=n_calibration_permutations,
            hidden_features=(64, 64),
        ).to(device)

    # Train the model
    train(
        n_nominal=n_nominal,
        nominal_observations=nominal_states,
        failure_guide=failure_guide,
        n_failure=n_failure,
        failure_observations=failure_states,
        n_failure_eval=n_failure_eval,
        failure_observations_eval=failure_eval_states,
        failure_posterior_samples_eval=None,
        nominal_posterior_samples_eval=None,
        objective_fn=objective_fn,
        divergence_fn=divergence_fn,
        plot_posterior=plot_posterior,
        plot_posterior_grid=plot_posterior_grid,
        name="swi/" + run_name,
        calibrate=calibrate,
        regularize=regularize,
        num_steps=n_steps,
        lr=lr,
        lr_gamma=lr_gamma,
        lr_steps=lr_steps,
        grad_clip=grad_clip,
        weight_decay=weight_decay,
        num_calibration_points=n_calibration_particles,
        calibration_weight=calibration_weight,
        regularization_weight=regularization_weight,
        elbo_weight=elbo_weight,
        wasserstein_regularization=wasserstein,
        calibration_num_permutations=n_calibration_permutations,
        calibration_ub=calibration_ub,
        calibration_lr=calibration_lr,
        calibration_substeps=calibration_substeps,
        calibration_steps=calibration_steps,
        # plot_every_n=n_steps,
        plot_every_n=10,
        device=device,
    )


if __name__ == "__main__":
    run()
