"""Implement CalVI training for the two moons toy problem."""
import os

import matplotlib
import matplotlib.pyplot as plt
import pyro
import torch
import zuko
from click import command, option

import wandb
from scripts.mnist.model import load_mnist_data
from scripts.training import train
from scripts.utils import kl_divergence


@command()
@option("--n-nominal", default=500, help="# of nominal examples")
@option("--n-failure", default=5, help="# of failure examples for training")
@option("--n-failure-eval", default=500, help="# of failure examples for evaluation")
@option("--no-calibrate", is_flag=True, help="Don't use calibration")
@option("--regularize", is_flag=True, help="Regularize failure using KL wrt nominal")
@option("--wasserstein", is_flag=True, help="Regularize failure using W2 wrt nominal")
@option("--seed", default=0, help="Random seed")
@option("--n-steps", default=1000, type=int, help="# of steps")
@option("--lr", default=1e-3, type=float, help="Learning rate")
@option("--lr-gamma", default=1.0, type=float, help="Learning rate decay")
@option("--lr-steps", default=1000, type=int, help="Steps per learning rate decay")
@option("--grad-clip", default=100, type=float, help="Gradient clipping value")
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
    default=1,
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
    exclude_nominal,
):
    """Generate data and train the SWI model."""
    matplotlib.use("Agg")
    matplotlib.rcParams["figure.dpi"] = 300

    # Parse arguments
    calibrate = not no_calibrate

    # Generate data (use consistent seed for all runs to make data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    pyro.set_rng_seed(0)

    # Generate training data
    with torch.no_grad():
        nominal_samples, failure_samples, failure_samples_eval = load_mnist_data(
            device, n_nominal, n_failure, n_failure_eval
        )

    # Change seed for training
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    # Make the objective and divergence closures
    def objective_fn(guide_dist, n, obs):
        """Compute the data likelihood."""
        data_likelihood = guide_dist.log_prob(obs).mean()

        return -data_likelihood  # negative because we want to maximize

    def divergence_fn(p, q):
        """Compute the KL divergence"""
        return kl_divergence(p, q, n_divergence_particles)

    # Define plotting callbacks
    @torch.no_grad()
    def plot_posterior(*dists, labels=None, save_file_name=None, save_wandb=False):
        fig, axs = plt.subplots(1, len(dists), figsize=(5 * len(dists), 5))
        for i, dist in enumerate(dists):
            sample = dist.sample().cpu().numpy().reshape(28, 28)
            axs[i].imshow(sample, cmap="gray")

            if labels is not None:
                axs[i].set_title(labels[i])

            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].axis("off")

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

                sample = failure_guide(label).sample().cpu().numpy().reshape(28, 28)
                axs[row, i].imshow(sample, cmap="gray")

                axs[row, i].set_xticks([])
                axs[row, i].set_yticks([])
                axs[row, i].axis("off")

        if save_file_name:
            plt.savefig(save_file_name, bbox_inches="tight", dpi=300)

        if save_wandb:
            wandb.log({"Posterior grid": wandb.Image(fig)}, commit=False)

        plt.close()

    # Start wandb
    run_name = run_prefix
    run_name += "ours_" if (calibrate and not regularize) else ""
    run_name += "calibrated_" if calibrate else "uncalibrated_"
    if regularize:
        run_name += "regularized_kl" if not wasserstein else "unregularized_w2"
    wandb.init(
        project="mnist-1",
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
            "calibration_steps": calibration_steps,
            "exclude_nominal": exclude_nominal,
        },
    )

    # Make a directory for checkpoints if it doesn't already exist
    os.makedirs(f"checkpoints/mnist/{run_name}", exist_ok=True)

    # Initialize the models
    if wasserstein:
        failure_guide = zuko.flows.CNF(
            features=28 * 28,
            context=n_calibration_permutations,
            hidden_features=(256, 256),
        ).to(device)
    else:
        failure_guide = zuko.flows.NSF(
            features=28 * 28,
            context=n_calibration_permutations,
            hidden_features=(256, 256),
        ).to(device)

    # Train the model
    train(
        n_nominal=n_nominal,
        nominal_observations=nominal_samples,
        failure_guide=failure_guide,
        n_failure=n_failure,
        failure_observations=failure_samples,
        n_failure_eval=n_failure_eval,
        failure_observations_eval=failure_samples_eval,
        failure_posterior_samples_eval=None,
        nominal_posterior_samples_eval=None,
        objective_fn=objective_fn,
        divergence_fn=divergence_fn,
        plot_posterior=plot_posterior,
        plot_posterior_grid=plot_posterior_grid,
        name="mnist/" + run_name,
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
        plot_every_n=100,
        exclude_nominal=exclude_nominal,
    )

    wandb.finish()


if __name__ == "__main__":
    run()
