"""Evaluate trained models."""

import pyro
import torch
import zuko

from scripts.swi.model import NX_COARSE, NY_COARSE, seismic_model
from scripts.swi.train import elbo_loss


def load_flow(flow_path, device):
    """Load a trained flow."""
    flow = zuko.flows.NSF(features=NX_COARSE * NY_COARSE, context=1).to(device)
    flow.load_state_dict(torch.load(flow_path))
    flow.eval()
    return flow


if __name__ == "__main__":
    seed = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    # Load a trained flow and plot the log probability calibration
    name = "swi_vi_ours_amortized_calibrated_unregularized"
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

    num_elbo_particles = 10
    num_divergence_particles = 10
    num_divergence_points = 10
    nominal_label = torch.tensor([0.0], device=device)
    failure_label = torch.tensor([1.0], device=device)
    divergence_bounds = torch.linspace(0.0, 1.0, num_divergence_points).to(device)
    nominal_context = torch.tensor([[0.0]] * num_divergence_points).to(device)

    # What is the derivative of the ELBO wrt the divergence bound?
    result = elbo_loss(
        seismic_model,
        guide,
        failure_label,
        num_elbo_particles,
        N=n_failure,
        receiver_observations=failure_observations,
    )
    elbos.append(-result[0])
    logprobs.append(result[1])
