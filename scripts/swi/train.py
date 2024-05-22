"""Implement CalVI training for the SWI problem."""

import os

import matplotlib
import matplotlib.pyplot as plt
import pyro
import torch
import zuko
from click import command, option

import wandb
from scripts.swi.model import NX_COARSE, NY_COARSE, seismic_model
from scripts.training import train
from scripts.utils import kl_divergence, ConditionalGaussianMixture


@command()
@option("--n-nominal", default=100, help="# of nominal examples")
@option("--n-failure", default=4, help="# of failure examples for training")
@option("--n-failure-eval", default=500, help="# of failure examples for evaluation")
@option("--no-calibrate", is_flag=True, help="Don't use calibration")
@option("--balance", is_flag=True, help="Balance CalNF")
@option("--bagged", is_flag=True, help="Bootstrap aggregation")
@option("--regularize", is_flag=True, help="Regularize failure using KL wrt nominal")
@option("--wasserstein", is_flag=True, help="Regularize failure using W2 wrt nominal")
@option("--gmm", is_flag=True, help="Use GMM instead of NF")
@option("--seed", default=0, help="Random seed")
@option("--n-steps", default=500, type=int, help="# of steps")
@option("--lr", default=1e-3, type=float, help="Learning rate")
@option("--lr-gamma", default=1.0, type=float, help="Learning rate decay")
@option("--lr-steps", default=1000, type=int, help="Steps per learning rate decay")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    # Generate training data
    with torch.no_grad():
        # Generate the data for the nominal and failure cases
        observation_noise_scale = 1e-2
        profile_background = torch.zeros(NY_COARSE, NX_COARSE, device=device)

        # Nominal has a layer of higher vp, vs, and rho in the middle
        profile_nominal = profile_background.expand(n_nominal, -1, -1).clone()
        profile_nominal[:, 3:6, 1:9] = 1.0
        # fracture_variation = 0.5 * torch.randn(n_nominal).to(device).reshape(-1, 1, 1)
        # profile_nominal[:, 3:6, 4:6] += fracture_variation
        # profile_nominal[:, 3:4, 6:10] += fracture_variation
        # profile_nominal[:, 3:6, :1] -= fracture_variation
        # profile_nominal[:, 4:7, 9:] -= fracture_variation
        profile_nominal += 0.3 * torch.randn_like(profile_nominal)

        nominal_observations = []
        for i in range(n_nominal):
            nominal_model = pyro.poutine.condition(
                seismic_model, data={"profile": profile_nominal[i]}
            )
            nominal_observations.append(
                nominal_model(
                    N=1,
                    observation_noise_scale=observation_noise_scale,
                    device=device,
                )
            )
        nominal_observations = torch.cat(nominal_observations)

        n_nominal_eval = n_failure_eval
        profile_nominal_eval = profile_background.expand(n_nominal_eval, -1, -1).clone()
        profile_nominal_eval[:, 3:6, 1:9] = 1.0
        profile_nominal_eval += 0.3 * torch.randn_like(profile_nominal_eval)

        nominal_observations_eval = []
        for i in range(n_nominal_eval):
            nominal_model = pyro.poutine.condition(
                seismic_model, data={"profile": profile_nominal_eval[i]}
            )
            nominal_observations_eval.append(
                nominal_model(
                    N=1, observation_noise_scale=observation_noise_scale, device=device
                )
            )
        nominal_observations_eval = torch.cat(nominal_observations_eval)

        # Failure has a break in the middle of the layer
        profile_failure = profile_background.expand(n_failure, -1, -1).clone()
        profile_failure[:, 3:6, 0:4] = 1.0
        profile_failure[:, 4:7, 6:10] = 1.0
        profile_failure += 0.3 * torch.randn_like(profile_failure)

        failure_observations = []
        for i in range(n_failure):
            failure_model = pyro.poutine.condition(
                seismic_model, data={"profile": profile_failure[i]}
            )
            failure_observations.append(
                failure_model(
                    N=1, observation_noise_scale=observation_noise_scale, device=device
                )
            )
        failure_observations = torch.cat(failure_observations)

        # Also generate samples for evaluation
        profile_failure_eval = profile_background.expand(n_failure_eval, -1, -1).clone()
        profile_failure_eval[:, 3:6, 0:4] = 1.0
        profile_failure_eval[:, 4:7, 6:10] = 1.0
        profile_failure_eval += 0.3 * torch.randn_like(profile_failure_eval)

        failure_observations_eval = []
        for i in range(n_failure_eval):
            failure_model = pyro.poutine.condition(
                seismic_model, data={"profile": profile_failure_eval[i]}
            )
            failure_observations_eval.append(
                failure_model(
                    N=1, observation_noise_scale=observation_noise_scale, device=device
                )
            )
        failure_observations_eval = torch.cat(failure_observations_eval)

    # Make the objective and divergence closures
    def single_particle_elbo(guide_dist, n, obs):
        posterior_sample, posterior_logprob = guide_dist.rsample_and_log_prob()

        # Reshape the sample
        posterior_sample = posterior_sample.reshape(NY_COARSE, NX_COARSE)

        model_trace = pyro.poutine.trace(
            pyro.poutine.condition(seismic_model, data={"profile": posterior_sample})
        ).get_trace(N=n, receiver_observations=obs, device=obs.device)
        model_logprob = model_trace.log_prob_sum()

        return model_logprob - posterior_logprob

    def objective_fn(guide_dist, n, obs):
        """ELBO loss for the seismic waveform inversion problem."""
        elbo = torch.tensor(0.0).to(obs.device)
        for _ in range(n_elbo_particles):
            elbo += single_particle_elbo(guide_dist, n, obs) / n_elbo_particles

        # Make it negative to make it a loss and scale by the dimension
        return -elbo / (NY_COARSE * NX_COARSE)

    def divergence_fn(p, q):
        """Compute the KL divergence"""
        return kl_divergence(p, q, n_divergence_particles)

    # Also make a closure for classifying anomalies
    def score_fn(nominal_guide_dist, failure_guide_dist, n, obs):
        scores = torch.zeros(n).to(obs.device)

        n_samples = 10
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
            scores[i] = failure_elbo * 1e-3

        return scores

    # Define plotting callbacks
    @torch.no_grad()
    def plot_posterior(*dists, labels=None, save_file_name=None, save_wandb=False):
        fig, axs = plt.subplots(len(dists), 1, figsize=(4, 4 * len(dists)))

        for i, dist in enumerate(dists):
            sample_mean = dist.sample((100,)).mean(dim=0).cpu().numpy()
            sample_mean = sample_mean.reshape(NY_COARSE, NX_COARSE)
            axs[i].imshow(sample_mean, cmap="Blues")

        if labels:
            for i, label in enumerate(labels):
                axs[i].set_ylabel(label)

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

                sample_mean = (
                    failure_guide(label).sample((100,)).mean(dim=0).cpu().numpy()
                )
                sample_mean = sample_mean.reshape(NY_COARSE, NX_COARSE)

                axs[row, i].imshow(sample_mean, cmap="Blues")

        if save_file_name:
            plt.savefig(save_file_name, bbox_inches="tight", dpi=300)

        if save_wandb:
            wandb.log({"Posterior grid": wandb.Image(fig)}, commit=False)

        plt.close()

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
        project=f"swi-{project_suffix}",  # TODO
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
    os.makedirs(f"checkpoints/swi/{run_name}_{seed}", exist_ok=True)

    # Initialize the models
    if wasserstein:
        failure_guide = zuko.flows.CNF(
            features=NY_COARSE * NX_COARSE,
            context=n_calibration_permutations,
            hidden_features=(64, 64),
        ).to(device)
    elif gmm:
        failure_guide = ConditionalGaussianMixture(
            n_context=n_calibration_permutations,
            n_features=NY_COARSE * NX_COARSE,
        ).to(device)
    else:
        failure_guide = zuko.flows.NSF(
            features=NY_COARSE * NX_COARSE,
            context=n_calibration_permutations,
            hidden_features=(64, 64),
        ).to(device)

    # Train the model
    train(
        n_nominal=n_nominal,
        nominal_observations=nominal_observations,
        failure_guide=failure_guide,
        n_failure=n_failure,
        failure_observations=failure_observations,
        n_nominal_eval=n_nominal_eval,
        nominal_observations_eval=nominal_observations_eval,
        n_failure_eval=n_failure_eval,
        failure_observations_eval=failure_observations_eval,
        failure_posterior_samples_eval=profile_failure_eval.reshape(
            -1, NY_COARSE * NX_COARSE
        ),
        nominal_posterior_samples_eval=profile_nominal.reshape(
            -1, NY_COARSE * NX_COARSE
        ),
        objective_fn=objective_fn,
        divergence_fn=divergence_fn,
        plot_posterior=plot_posterior,
        plot_posterior_grid=plot_posterior_grid,
        name="swi/" + run_name + f"_{seed}",
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
        plot_every_n=n_steps,
        exclude_nominal=exclude_nominal,
        score_fn=score_fn,
        balance=balance,
        bagged=bagged,
    )


if __name__ == "__main__":
    run()
