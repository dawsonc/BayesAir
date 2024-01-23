"""Evaluate trained models."""

import pyro
import torch
import zuko

from scripts.swi.model import NX_COARSE, NY_COARSE, seismic_model
from scripts.swi.train import plot_posterior, plot_posterior_interp


def load_flow(flow_path, device):
    """Load a trained flow."""
    flow = zuko.flows.NSF(features=NX_COARSE * NY_COARSE, context=1).to(device)
    flow.load_state_dict(torch.load(flow_path))
    flow.eval()
    return flow


@torch.no_grad()
def eval(name):
    seed = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    # Load a trained flow and plot the log probability calibration
    flow_path = "checkpoints/swi/" + name + "/checkpoint_299.pth"
    guide = load_flow(flow_path, device=device)

    # Make a bunch of eval data
    n_failure = 100
    with torch.no_grad():
        profile_background = torch.zeros(NY_COARSE, NX_COARSE, device=device)

        # Failure has a break in the middle of the layer
        profile_failure = profile_background.expand(n_failure, -1, -1).clone()
        profile_failure[:, 3:6, 0:4] = 1.0
        profile_failure[:, 4:7, 6:10] = 1.0
        profile_failure += 0.2 * torch.randn_like(profile_failure)

        # Generate the data for the nominal and failure cases
        observation_noise_scale = 1e-2

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

    num_elbo_particles = 50
    num_divergence_particles = 100
    num_divergence_points = 10
    nominal_label = torch.tensor([0.0], device=device)
    failure_label = torch.tensor([1.0], device=device)
    divergence_bounds = torch.linspace(0.0, 1.0, num_divergence_points).to(device)
    nominal_context = torch.tensor([[0.0]] * num_divergence_points).to(device)

    # # Get the divergences
    # print("Computing divergences...")
    # divergences = kl_divergence(
    #     guide,
    #     guide,
    #     divergence_bounds.reshape(-1, 1),
    #     nominal_context,
    #     num_divergence_particles,
    # )
    # failure_divergence = divergences[-1]
    # print("Done.")

    # print("Plotting calibration...")
    # plot_logprob_calibration(
    #     guide,
    #     failure_label,
    #     num_elbo_particles,
    #     n_failure,
    #     failure_observations,
    #     divergence_bounds,
    #     failure_divergence,
    #     divergences,
    #     save_file_name="eval_" + name + "_calibration_logprob_evidence.png",
    #     save_wandb=False,
    # )
    # print("Done.")

    print("Plotting posterior...")
    plot_posterior(
        guide,
        guide,
        nominal_label,
        failure_label,
        save_file_name="eval_" + name + "_posterior.png",
        samples=1000,
        save_wandb=False,
    )
    plot_posterior_interp(
        guide,
        divergence_bounds,
        save_file_name="eval_" + name + "_posterior_interpolated.png",
        samples=1000,
        save_wandb=False,
    )
    print("Done.")


if __name__ == "__main__":
    names = [
        "swi_vi_ours_amortized_calibrated_unregularized",  # ours
        "swi_vi_amortized_uncalibrated_regularized_kl_beta_0.10",  # KL regularized
        "swi_vi_amortized_uncalibrated_regularized_kl_beta_0.01",  # KL regularized
        "swi_vi_amortized_uncalibrated_regularized_kl_beta_1.00",  # KL regularized
    ]
    for name in names:
        eval(name)
