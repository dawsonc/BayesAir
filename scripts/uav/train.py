"""Implement CalVI training for the UAV problem."""

import os

import matplotlib
import matplotlib.pyplot as plt
import pyro
import torch
import zuko
from click import command, option

import wandb
from scripts.training import train
from scripts.uav.data import load_all_data
from scripts.uav.model import model as uav_model
from scripts.utils import kl_divergence, ConditionalGaussianMixture


@command()
@option("--n-nominal", default=10, help="# of nominal examples")
@option("--n-failure", default=1, help="# of failure examples for training")
@option("--n-failure-eval", default=1, help="# of failure examples for evaluation")
@option("--no-calibrate", is_flag=True, help="Don't use calibration")
@option("--balance", is_flag=True, help="Balance CalNF")
@option("--bagged", is_flag=True, help="Bootstrap aggregation")
@option("--regularize", is_flag=True, help="Regularize failure using KL wrt nominal")
@option("--ablation", is_flag=True, help="If true, set project label to ablation")
@option("--wasserstein", is_flag=True, help="Regularize failure using W2 wrt nominal")
@option("--gmm", is_flag=True, help="Use GMM instead of NF")
@option("--seed", default=0, help="Random seed")
@option("--n-steps", default=500, type=int, help="# of steps")
@option("--lr", default=1e-2, type=float, help="Learning rate")
@option("--lr-gamma", default=0.1, type=float, help="Learning rate decay")
@option("--lr-steps", default=250, type=int, help="Steps per learning rate decay")
@option("--grad-clip", default=10, type=float, help="Gradient clipping value")
@option("--weight-decay", default=0.0, type=float, help="Weight decay rate")
@option("--run-prefix", default="", help="Prefix for run name")
@option("--project-suffix", default="benchmark", help="Suffix for project name")
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
    default=6,
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
    "--exclude-nominal",
    is_flag=True,
    help="If True, don't learn the nominal distribution",
)
def run(
    n_nominal,
    n_failure,
    n_failure_eval,
    no_calibrate,
    balance,
    bagged,
    regularize,
    ablation,
    wasserstein,
    gmm,
    seed,
    n_steps,
    lr,
    lr_gamma,
    lr_steps,
    grad_clip,
    weight_decay,
    run_prefix,
    project_suffix,
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
    exclude_nominal,
):
    """Generate data and train the SWI model."""
    matplotlib.use("Agg")
    matplotlib.rcParams["figure.dpi"] = 300

    # Parse arguments
    calibrate = not no_calibrate

    # Generate data (use the same seed for all runs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    pyro.set_rng_seed(0)

    # Load training and eval data
    with torch.no_grad():
        nominal_data, _, rudder_data = load_all_data()  # elevator failure data not used

        # Unpack nominal data into tensors (concatenate trajectories)
        (
            _,  # time unused
            nominal_initial_states,
            nominal_next_states,
            nominal_pqrs,
            nominal_commands,
        ) = nominal_data
        nominal_initial_states_train = torch.cat(nominal_initial_states[1:])
        nominal_next_states_train = torch.cat(nominal_next_states[1:])
        nominal_pqrs_train = torch.cat(nominal_pqrs[1:])
        nominal_commands_train = torch.cat(nominal_commands[1:])

        # These are all N x 3, so stack into N x 4 x 3
        nominal_observations = torch.stack(
            (
                nominal_initial_states_train,
                nominal_next_states_train,
                nominal_pqrs_train,
                nominal_commands_train,
            ),
            dim=1,
        )
        n_nominal = nominal_observations.shape[0]

        nominal_initial_states_eval = torch.cat(nominal_initial_states[:1])
        nominal_next_states_eval = torch.cat(nominal_next_states[:1])
        nominal_pqrs_eval = torch.cat(nominal_pqrs[:1])
        nominal_commands_eval = torch.cat(nominal_commands[:1])

        # These are all N x 3, so stack into N x 4 x 3
        nominal_observations_eval = torch.stack(
            (
                nominal_initial_states_eval,
                nominal_next_states_eval,
                nominal_pqrs_eval,
                nominal_commands_eval,
            ),
            dim=1,
        )
        n_nominal_eval = nominal_observations_eval.shape[0]

        # Unpack failure data into tensors (concatenate trajectories)
        (
            _,
            failure_initial_states,
            failure_next_states,
            failure_pqrs,
            failure_commands,
        ) = rudder_data
        failure_initial_states_train = torch.cat(failure_initial_states[:1])
        failure_next_states_train = torch.cat(failure_next_states[:1])
        failure_pqrs_train = torch.cat(failure_pqrs[:1])
        failure_commands_train = torch.cat(failure_commands[:1])
        failure_observations_train = torch.stack(
            (
                failure_initial_states_train,
                failure_next_states_train,
                failure_pqrs_train,
                failure_commands_train,
            ),
            dim=1,
        )
        n_failure = failure_observations_train.shape[0]

        failure_initial_states_eval = torch.cat(failure_initial_states[1:])
        failure_next_states_eval = torch.cat(failure_next_states[1:])
        failure_pqrs_eval = torch.cat(failure_pqrs[1:])
        failure_commands_eval = torch.cat(failure_commands[1:])
        failure_observations_eval = torch.stack(
            (
                failure_initial_states_eval,
                failure_next_states_eval,
                failure_pqrs_eval,
                failure_commands_eval,
            ),
            dim=1,
        )
        n_failure_eval = failure_observations_eval.shape[0]

    # Vary the seed for training
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    def single_particle_elbo(guide_dist, n, obs):
        posterior_sample, posterior_logprob = guide_dist.rsample_and_log_prob()

        # Reshape the sample
        A = posterior_sample[:9].reshape(3, 3)
        K = posterior_sample[9:18].reshape(3, 3)
        d = posterior_sample[18:21].reshape(3)
        log_noise_strength = posterior_sample[21]

        # Unpack the observations
        initial_states = obs[:, 0]
        next_states = obs[:, 1]
        pqrs = obs[:, 2]
        commands = obs[:, 3]

        model_trace = pyro.poutine.trace(
            pyro.poutine.condition(
                uav_model,
                data={
                    "A": A,
                    "K": K,
                    "d": d,
                    "log_noise_strength": log_noise_strength,
                },
            )
        ).get_trace(
            initial_states=initial_states,
            commands=commands,
            observed_next_states=next_states,
            observed_pqrs=pqrs,
            dt=0.25,
            device=obs.device,
        )
        model_logprob = model_trace.log_prob_sum()

        return model_logprob - posterior_logprob

    # Make the objective and divergence closures
    def objective_fn(guide_dist, n, obs):
        """ELBO loss for the seismic waveform inversion problem."""
        elbo = torch.tensor(0.0).to(obs.device)
        for _ in range(n_elbo_particles):
            elbo += single_particle_elbo(guide_dist, n, obs) / n_elbo_particles

        # Make it negative to make it a loss
        return -elbo

    def divergence_fn(p, q):
        """Compute the KL divergence"""
        return kl_divergence(p, q, n_divergence_particles)

    # Also make a closure for classifying anomalies
    def score_fn(nominal_guide_dist, failure_guide_dist, n, obs):
        scores = torch.zeros(n).to(obs.device)

        n_samples = 50
        for i in range(n):
            # nominal_elbo = torch.tensor(0.0).to(obs.device)
            # for _ in range(n_samples):
            #     nominal_elbo += (
            #         single_particle_elbo(nominal_guide_dist, 1, obs[i].unsqueeze(0))
            #         / n_samples
            #     )

            failure_elbo = torch.tensor(0.0).to(obs.device)
            for _ in range(n_samples):
                failure_elbo += (
                    single_particle_elbo(failure_guide_dist, 1, obs[i].unsqueeze(0))
                    / n_samples
                )

            # scores[i] = failure_elbo - nominal_elbo
            scores[i] = failure_elbo

        return scores

    # Define plotting callbacks
    @torch.no_grad()
    def plot_posterior(*dists, labels=None, save_file_name=None, save_wandb=False):
        fig, axs = plt.subplots(
            3, len(dists), figsize=(4 * len(dists), 4 * 3), sharey=True
        )

        assert (
            len(dists) == 2
        ), "UAV plot_posterior assumes only nominal and failure dists"

        for i, (dist, obs) in enumerate(
            zip(dists, (nominal_observations, failure_observations_train))
        ):
            (initial_states, _, pqr_obs, commands) = obs.transpose(0, 1)

            posterior_sample = dist.sample((100,)).mean(dim=0)
            A = posterior_sample[:9].reshape(3, 3)
            K = posterior_sample[9:18].reshape(3, 3)
            d = posterior_sample[18:21].reshape(3, 1)
            states = initial_states.reshape(-1, 3, 1)
            commands = commands.reshape(-1, 3, 1)
            errors = commands - states
            pqr_pred = A @ states + K @ errors + d

            for row in range(3):
                var = ["p", "q", "r"][row]
                axs[row, i].plot(pqr_obs[:, row].cpu(), "b-", label="Observed " + var)
                axs[row, i].plot(
                    pqr_pred[:, row].cpu(), "r--", label="Predicted " + var
                )

                if i == 0:
                    axs[row, i].set_ylabel(var)

                if row == 0:
                    axs[row, i].set_title(["Nominal", "Failure"][i])

                if row == 0 and i == 1:
                    axs[row, i].legend()

        fig.tight_layout()

        if save_file_name:
            plt.savefig(save_file_name, bbox_inches="tight", dpi=300)

        if save_wandb:
            wandb.log({"Posteriors": wandb.Image(fig)}, commit=False)

        plt.close()

    @torch.no_grad()
    def plot_posterior_grid(
        failure_guide,
        nominal_label,
        save_file_name=None,
        save_wandb=False,
    ):
        pass  # Not implemented -- what would be useful to plot here?

    # Start wandb
    run_name = run_prefix
    run_name += "ours_" if (calibrate and not regularize) else ""
    run_name += "balanced_" if balance else ""
    run_name += "bagged_" if bagged else ""
    run_name += "gmm_" if gmm else ""
    run_name += "calibrated_" if calibrate else ""
    if regularize:
        run_name += "kl_regularized_kl" if not wasserstein else "w2_regularized"
    wandb.init(
        project=f"uav-{project_suffix}",
        name=run_name,
        group=run_name,
        config={
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
            "exclude_nominal": exclude_nominal,
        },
    )

    # Make a directory for checkpoints if it doesn't already exist
    os.makedirs(f"checkpoints/uav/{run_name}_{seed}", exist_ok=True)

    # Initialize the models
    if wasserstein:
        failure_guide = zuko.flows.CNF(
            features=2 * 3 * 3 + 3 + 1,
            context=n_calibration_permutations,
            hidden_features=(64, 64),
        ).to(device)
    elif gmm:
        failure_guide = ConditionalGaussianMixture(
            n_context=n_calibration_permutations,
            n_features=2 * 3 * 3 + 3 + 1,
        ).to(device)
    else:
        failure_guide = zuko.flows.NSF(
            features=2 * 3 * 3 + 3 + 1,
            context=n_calibration_permutations,
            hidden_features=(64, 64),
        ).to(device)

    # Train the model
    train(
        n_nominal=n_nominal,
        nominal_observations=nominal_observations,
        failure_guide=failure_guide,
        n_failure=n_failure,
        failure_observations=failure_observations_train,
        n_nominal_eval=n_nominal_eval,
        nominal_observations_eval=nominal_observations_eval,
        n_failure_eval=n_failure_eval,
        failure_observations_eval=failure_observations_eval,
        failure_posterior_samples_eval=None,
        nominal_posterior_samples_eval=None,
        objective_fn=objective_fn,
        divergence_fn=divergence_fn,
        plot_posterior=plot_posterior,
        plot_posterior_grid=plot_posterior_grid,
        name="uav/" + run_name + f"_{seed}",
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
        # plot_every_n=n_steps,
        plot_every_n=100,
        exclude_nominal=exclude_nominal,
        score_fn=score_fn,
        balance=balance,
        bagged=bagged,
    )


if __name__ == "__main__":
    run()
