"""Implement CalVI training for the seismic waveform model."""
import os

import click
import matplotlib
import matplotlib.pyplot as plt
import pyro
import torch
import zuko
from tqdm import tqdm

import wandb
from scripts.seismic.model import NX_COARSE, NY_COARSE, seismic_model
from scripts.utils import kl_divergence


def elbo_loss(model, guide, context, num_particles=10, *model_args, **model_kwargs):
    """ELBO loss for the seismic waveform inversion problem."""
    elbo = torch.tensor(0.0).to(context.device)
    logprob = torch.tensor(0.0).to(context.device)
    guide_dist = guide(context)
    for _ in range(num_particles):
        posterior_sample, posterior_logprob = guide_dist.rsample_and_log_prob()

        # Reshape the sample
        posterior_sample = posterior_sample.reshape(NY_COARSE, NX_COARSE)

        model_trace = pyro.poutine.trace(
            pyro.poutine.condition(model, data={"profile": posterior_sample})
        ).get_trace(*model_args, **model_kwargs, device=context.device)
        model_logprob = model_trace.log_prob_sum()

        elbo += (model_logprob - posterior_logprob) / num_particles
        logprob += model_logprob / num_particles

    return (
        -elbo,  #  negative to make it a loss
        logprob,
    )


def plot_posterior(
    nominal_guide,
    failure_guide,
    nominal_label,
    failure_label,
    samples=10,
    save_file_name=None,
    save_wandb=True,
):
    """Plot the posterior distributions for the nominal and failure cases."""
    fig, axs = plt.subplots(2, 1, figsize=(4, 8))

    # Sample from the posteriors
    with torch.no_grad():
        nominal_posterior = (
            nominal_guide(nominal_label).sample((samples,)).mean(dim=0).cpu().numpy()
        )
        nominal_posterior = nominal_posterior.reshape(NY_COARSE, NX_COARSE)

        failure_posterior = (
            failure_guide(failure_label).sample((samples,)).mean(dim=0).cpu().numpy()
        )
        failure_posterior = failure_posterior.reshape(NY_COARSE, NX_COARSE)

    axs[0].imshow(nominal_posterior, cmap="Greys")
    axs[1].imshow(failure_posterior, cmap="Greys")
    fig.tight_layout()

    if save_file_name is not None:
        plt.savefig("paper_plots/swi/" + save_file_name, dpi=1000)

    if save_wandb:
        wandb.log({"Posteriors": wandb.Image(fig)}, commit=False)

    plt.close()


def plot_posterior_interp(
    failure_guide,
    failure_label,
    samples=10,
    n_steps=10,
    save_file_name=None,
    save_wandb=True,
):
    """Plot the interpolated posterior distribution between the two cases."""
    n_steps = 10
    steps = torch.linspace(0.0, 1.0, n_steps, device=failure_label.device)
    fig, axs = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4))
    with torch.no_grad():
        for idx, label in enumerate(steps):
            axs[idx].imshow(
                failure_guide(label.reshape(1))
                .sample((samples,))
                .mean(dim=0)
                .cpu()
                .numpy()
                .reshape(NY_COARSE, NX_COARSE),
                cmap="Greys",
            )
            axs[idx].set_title(f"Label {label}")
            axs[idx].axis("off")

    if save_file_name is not None:
        plt.savefig("paper_plots/swi/" + save_file_name, dpi=1000)

    if save_wandb:
        wandb.log({"Interpolated posteriors": wandb.Image(fig)}, commit=False)

    plt.close()


def plot_label_calibration(
    divergence_bounds,
    failure_divergence,
    divergences,
    save_file_name=None,
    save_wandb=True,
):
    """Plot the KL calibration curve."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    x = torch.linspace(0.0, 1.0, divergence_bounds.shape[0])
    ax.plot(
        x,
        divergence_bounds.cpu() * failure_divergence.detach().cpu(),
        "k--",
        label="Bound",
    )
    ax.plot(
        x,
        divergences.detach().cpu(),
        "b-",
        label="Measured",
    )
    ax.set_ylabel("KL Divergence")
    ax.set_xlabel("Label")
    fig.tight_layout()

    if save_file_name is not None:
        plt.savefig("paper_plots/swi/" + save_file_name, dpi=1000)

    if save_wandb:
        wandb.log({"Calibration": wandb.Image(fig)}, commit=False)

    plt.close()


def plot_logprob_calibration(
    failure_guide,
    failure_label,
    num_elbo_particles,
    n_failure,
    failure_observations,
    divergence_bounds,
    failure_divergence,
    divergences,
    save_file_name=None,
    save_wandb=True,
):
    """Plot the KL calibration curve."""
    fig, ax = plt.subplots(2, 1, figsize=(4, 8))

    with torch.no_grad():
        logprobs = []
        elbos = []
        for x in torch.linspace(
            0.0, 1.0, divergence_bounds.shape[0], device=failure_label.device
        ):
            print("Computing logprob for label", x)
            result = elbo_loss(
                seismic_model,
                failure_guide,
                x.reshape(1),
                num_elbo_particles,
                N=n_failure,
                receiver_observations=failure_observations,
            )
            elbos.append(-result[0])
            logprobs.append(result[1])

        logprobs = torch.stack(logprobs).reshape(-1)
        elbos = torch.stack(elbos).reshape(-1)

    # First plot shows KL divergence vs evidence
    ax[0].plot(
        divergences.detach().cpu(),
        logprobs.detach().cpu(),
        "bo",
        label="Measured",
    )
    ax[0].set_xlabel("KL Divergence")
    ax[0].set_ylabel("Failure Log Probability")

    # Second plot shows KL divergence vs ELBO
    ax[1].plot(
        divergences.detach().cpu(),
        elbos.detach().cpu(),
        "bo",
        label="Measured",
    )
    ax[1].set_xlabel("KL Divergence")
    ax[1].set_ylabel("Failure ELBO")

    fig.tight_layout()

    if save_file_name is not None:
        plt.savefig("paper_plots/swi/" + save_file_name, dpi=1000)

    if save_wandb:
        wandb.log({"Calibration Logprob and ELBO": wandb.Image(fig)}, commit=False)

    plt.close()


def train(
    n_nominal,
    nominal_observations,
    n_failure,
    failure_observations,
    name,
    amortize,
    calibrate,
    regularize,
    num_steps,
    lr,
    lr_gamma,
    lr_steps,
    grad_clip,
    weight_decay,
    num_elbo_particles,
    num_divergence_particles,
    num_divergence_points,
    divergence_weight,
    elbo_weight,
):
    """
    Compute the loss for the seismic waveform inversion problem.

    Args:
        n_nominal: Number of nominal observations.
        nominal_observations: Observed data for the nominal case.
        n_failure: Number of failure observations.
        failure_observations: Observed data for the failure case.
        name: the name for this run
        amortize: If true, learns one guide for both cases; otherwise, learns two
            separate guides.
        calibrate: If true, uses KL calibration
        regularize: If true, regularizes the failure case using the nominal case
        num_steps: number of optimization steps
        lr: learning rate
        lr_gamma: learning rate decay parameter
        lr_steps: number of steps between learning rate decays
        grad_clip: maximum gradient norm
        weight_decay: weight decay parameter
        num_elbo_particles: number of particles for ELBO estimation
        num_divergence_particles: number of particles for divergence estimation
        num_divergence_points: number of points for divergence calibration
        divergence_weight: weight applied to calibration loss
        elbo_weight: weight applied to ELBO loss
    """
    device = nominal_observations.device

    # Create the guide (represented using a normalizing flow)
    flow = zuko.flows.NSF(features=NX_COARSE * NY_COARSE, context=1).to(device)

    # If we are amortizing, we learn a single guide for both cases
    nominal_guide = flow
    failure_guide = flow
    # Otherwise, learn a second guide for the failure case
    if not amortize:
        failure_guide = zuko.flows.NSF(features=NX_COARSE * NY_COARSE, context=1).to(
            device
        )

    # Set up the optimizer
    if amortize:
        params = list(nominal_guide.parameters())
    else:
        params = list(nominal_guide.parameters()) + list(failure_guide.parameters())

    optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=lr_steps, gamma=lr_gamma
    )

    # Create the labels for training
    nominal_label = torch.tensor([0.0], device=device)
    failure_label = torch.tensor([1.0], device=device)
    divergence_bounds = torch.linspace(0.0, 1.0, num_divergence_points).to(device)
    nominal_context = torch.tensor([[0.0]] * num_divergence_points).to(device)

    # Train the model
    pbar = tqdm(range(num_steps))
    for i in pbar:
        optim.zero_grad()

        # Compute the loss components
        loss_components = {
            "nominal_elbo": elbo_loss(
                seismic_model,
                nominal_guide,
                nominal_label,
                num_elbo_particles,
                N=n_nominal,
                receiver_observations=nominal_observations,
            ),
            "failure_elbo": elbo_loss(
                seismic_model,
                failure_guide,
                failure_label,
                num_elbo_particles,
                N=n_failure,
                receiver_observations=failure_observations,
            ),
            "divergence": kl_divergence(
                failure_guide,
                nominal_guide,
                divergence_bounds.reshape(-1, 1),
                nominal_context,
                num_divergence_particles,
            ),
        }

        # Re-scale the divergence to [0, 1] (since we're using unbounded KL divergence)
        # Use the divergence at label=1 as the reference point
        failure_divergence = loss_components["divergence"][-1]

        # Compute the calibration loss based on the elbo and the deviation from
        # divergence bounds
        loss_components["divergence_deviation"] = (
            (
                loss_components["divergence"]
                / (loss_components["divergence"].max() + 1e-3)
                - divergence_bounds
            )
            ** 2
        ).mean()

        # Compute the loss
        loss = torch.tensor(0.0).to(device)
        loss += (
            elbo_weight * loss_components["nominal_elbo"][0] / (NY_COARSE * NX_COARSE)
        )
        loss += (
            elbo_weight * loss_components["failure_elbo"][0] / (NY_COARSE * NX_COARSE)
        )
        loss += (
            divergence_weight
            * loss_components["divergence_deviation"]
            * (1.0 if calibrate else 0.0)
        )

        if regularize:
            loss += divergence_weight * failure_divergence
            # samples = failure_guide(failure_label).rsample((num_divergence_particles,))
            # nominal_logprob = nominal_guide(nominal_label).log_prob(samples).mean()
            # loss -= divergence_weight * nominal_logprob / (NX_COARSE * NY_COARSE)

        # Step the optimizer
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optim.step()
        scheduler.step()

        # Record progress
        if i % 10 == 0 or i == num_steps - 1:
            plot_label_calibration(
                divergence_bounds,
                failure_divergence,
                loss_components["divergence"],
                save_file_name=name + "_calibration.png",
            )
            plot_logprob_calibration(
                failure_guide,
                failure_label,
                num_elbo_particles,
                n_failure,
                failure_observations,
                divergence_bounds,
                failure_divergence,
                loss_components["divergence"],
                save_file_name=name + "_calibration_logprob_evidence.png",
            )
            plot_posterior(
                nominal_guide,
                failure_guide,
                nominal_label,
                failure_label,
                save_file_name=name + "_posterior.png",
            )
            plot_posterior_interp(
                failure_guide,
                failure_label,
                save_file_name=name + "_posterior_interpolated.png",
            )
            if amortize:
                torch.save(
                    flow.state_dict(), f"checkpoints/swi/{name}/checkpoint_{i}.pth"
                )
            else:
                torch.save(
                    nominal_guide.state_dict(),
                    f"checkpoints/swi/{name}/nominal_checkpoint_{i}.pth",
                )
                torch.save(
                    failure_guide.state_dict(),
                    f"checkpoints/swi/{name}/failure_checkpoint_{i}.pth",
                )

        wandb.log(
            {
                "Loss": loss.detach().cpu().item(),
                "Nominal ELBO": loss_components["nominal_elbo"][0].detach().cpu().item()
                / (n_nominal * NY_COARSE * NX_COARSE),
                "Nominal log likelihood": loss_components["nominal_elbo"][1]
                .detach()
                .cpu()
                .item()
                / (n_nominal * NY_COARSE * NX_COARSE),
                "Failure ELBO": loss_components["failure_elbo"][0].detach().cpu().item()
                / (n_failure * NY_COARSE * NX_COARSE),
                "Failure log likelihood": loss_components["failure_elbo"][1]
                .detach()
                .cpu()
                .item()
                / (n_failure * NY_COARSE * NX_COARSE),
                "Calibration loss": loss_components["divergence_deviation"]
                .detach()
                .cpu()
                .item(),
                "Gradient norm": grad_norm.detach().cpu().item(),
            }
        )
        pbar.set_description(f"Loss: {loss.item():.4f}")


@click.command()
@click.option("--N-nominal", default=25, help="Number of nominal examples")
@click.option("--N-failure", default=3, help="Number of failure examples")
@click.option("--no-calibrate", is_flag=True, help="Don't use KL calibration")
@click.option("--no-amortize", is_flag=True, help="Don't amortize the guide")
@click.option("--regularize", is_flag=True, help="Regularize failure using nominal")
@click.option("--seed", default=0, help="Random seed")
@click.option("--num-steps", default=300, type=int, help="Number of steps")
@click.option("--lr", default=1e-2, type=float, help="Learning rate")
@click.option("--lr-gamma", default=0.1, type=float, help="Learning rate decay")
@click.option("--lr-steps", default=200, type=int, help="Steps per learning rate decay")
@click.option("--grad-clip", default=10.0, type=float, help="Gradient clipping value")
@click.option("--weight-decay", default=1e-4, type=float, help="Weight decay rate")
@click.option(
    "--num-elbo-particles",
    default=3,
    type=int,
    help="number of particles for ELBO estimation",
)
@click.option(
    "--num-divergence-particles",
    default=10,
    type=int,
    help="number of particles for divergence estimation",
)
@click.option(
    "--num-divergence-points",
    default=10,
    type=int,
    help="number of points for divergence calibration",
)
@click.option(
    "--divergence-weight",
    default=1e0,
    type=float,
    help="weight applied to calibration loss",
)
@click.option(
    "--elbo-weight", default=1e0, type=float, help="weight applied to ELBO loss"
)
def run(
    n_nominal,
    n_failure,
    no_calibrate,
    no_amortize,
    regularize,
    seed,
    num_steps,
    lr,
    lr_gamma,
    lr_steps,
    grad_clip,
    weight_decay,
    num_elbo_particles,
    num_divergence_particles,
    num_divergence_points,
    divergence_weight,
    elbo_weight,
):
    """Generate data and train the SWI model."""
    matplotlib.use("Agg")

    # Parse arguments
    calibrate = not no_calibrate
    amortize = not no_amortize

    # Generate data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    with torch.no_grad():
        profile_background = torch.zeros(NY_COARSE, NX_COARSE, device=device)

        # Nominal has a layer of higher vp, vs, and rho in the middle
        profile_nominal = profile_background.expand(n_nominal, -1, -1).clone()
        profile_nominal[:, 3:6, 1:9] = 1.0
        profile_nominal += 0.2 * torch.randn_like(profile_nominal)

        # Failure has a break in the middle of the layer
        profile_failure = profile_background.expand(n_failure, -1, -1).clone()
        profile_failure[:, 3:6, 0:4] = 1.0
        profile_failure[:, 4:7, 6:10] = 1.0
        profile_failure += 0.2 * torch.randn_like(profile_failure)

        # Generate the data for the nominal and failure cases
        observation_noise_scale = 1e-2

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

    # Start wandb
    run_name = "swi_vi"
    run_name += "_ours" if (amortize and calibrate and not regularize) else ""
    run_name += "_amortized" if amortize else "_unamortized"
    run_name += "_calibrated" if calibrate else "_uncalibrated"
    run_name += "_regularized_kl" if regularize else "_unregularized"
    wandb.init(
        project="swi",
        name=run_name,
        group=run_name,
        config={
            "n_nominal": n_nominal,
            "n_failure": n_failure,
            "calibrate": calibrate,
            "amortize": amortize,
            "regularize": regularize,
            "seed": seed,
            "num_steps": num_steps,
            "lr": lr,
            "lr_gamma": lr_gamma,
            "lr_steps": lr_steps,
            "grad_clip": grad_clip,
            "weight_decay": weight_decay,
            "num_elbo_particles": num_elbo_particles,
            "num_divergence_particles": num_divergence_particles,
            "num_divergence_points": num_divergence_points,
            "divergence_weight": divergence_weight,
            "elbo_weight": elbo_weight,
        },
    )

    # Make a directory for checkpoints if it doesn't already exist
    os.makedirs(f"checkpoints/swi/{run_name}", exist_ok=True)

    # Train the model
    train(
        n_nominal,
        nominal_observations,
        n_failure,
        failure_observations,
        name=run_name,
        amortize=amortize,
        calibrate=calibrate,
        regularize=regularize,
        num_steps=num_steps,
        lr=lr,
        lr_gamma=lr_gamma,
        lr_steps=lr_steps,
        grad_clip=grad_clip,
        weight_decay=weight_decay,
        num_elbo_particles=num_elbo_particles,
        num_divergence_particles=num_divergence_particles,
        num_divergence_points=num_divergence_points,
        divergence_weight=divergence_weight,
        elbo_weight=elbo_weight,
    )


if __name__ == "__main__":
    run()
