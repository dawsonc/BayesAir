"""Implement the seismic waveform model."""
import deepwave
import pyro
import pyro.distributions as dist
import torch
from deepwave import elastic

# Define model parameters
NY = 30
NX = 100
DX = 4.0
N_SHOTS = 8
N_SOURCES_PER_SHOT = 1
D_SOURCE = 12
FIRST_SOURCE = 8
SOURCE_DEPTH = 2
N_RECEIVERS_PER_SHOT = 9
D_RECEIVER = 10
FIRST_RECEIVER = 0
RECEIVER_DEPTH = 2
FREQ = 15
NT = 100
DT = 0.004
PEAK_TIME = 1.5 / FREQ

# Use a coarser grid for the model than the simulation and normalize the latent params
X_DOWNSAMPLE = 10
Y_DOWNSAMPLE = 3
NX_COARSE = NX // X_DOWNSAMPLE
NY_COARSE = NY // Y_DOWNSAMPLE

VP_CENTER = 1500
VP_SCALE = 100
VS_CENTER = 1000
VS_SCALE = 100
RHO_CENTER = 2200
RHO_SCALE = 100


def seismic_model(
    N, receiver_observations=None, observation_noise_scale=1e-1, device=None
):
    # Set device automatically if not specified
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # source_locations
    source_locations = torch.zeros(
        N_SHOTS, N_SOURCES_PER_SHOT, 2, dtype=torch.long, device=device
    )
    source_locations[..., 0] = SOURCE_DEPTH
    source_locations[:, 0, 1] = torch.arange(N_SHOTS) * D_SOURCE + FIRST_SOURCE

    # receiver_locations
    receiver_locations = torch.zeros(
        N_SHOTS, N_RECEIVERS_PER_SHOT, 2, dtype=torch.long, device=device
    )
    receiver_locations[..., 0] = RECEIVER_DEPTH
    receiver_locations[:, :, 1] = (
        torch.arange(N_RECEIVERS_PER_SHOT) * D_RECEIVER + FIRST_RECEIVER
    ).repeat(N_SHOTS, 1)

    # source_amplitudes
    source_amplitudes = (
        (deepwave.wavelets.ricker(FREQ, NT, DT, PEAK_TIME))
        .repeat(N_SHOTS, N_SOURCES_PER_SHOT, 1)
        .to(device)
    )

    # Define generic priors on a normalized basis
    profile = pyro.sample(
        "profile",
        dist.Normal(
            torch.zeros(NY_COARSE, NX_COARSE, device=device),
            2.5 * torch.ones(NY_COARSE, NX_COARSE, device=device),
        ),
    )
    vp = profile * VP_SCALE + VP_CENTER
    vs = profile * VS_SCALE + VS_CENTER
    rho = profile * RHO_SCALE + RHO_CENTER

    # Upsample the model to the simulation grid
    vp = vp.repeat_interleave(Y_DOWNSAMPLE, dim=0).repeat_interleave(
        X_DOWNSAMPLE, dim=1
    )
    vs = vs.repeat_interleave(Y_DOWNSAMPLE, dim=0).repeat_interleave(
        X_DOWNSAMPLE, dim=1
    )
    rho = rho.repeat_interleave(Y_DOWNSAMPLE, dim=0).repeat_interleave(
        X_DOWNSAMPLE, dim=1
    )

    # Run the simulation
    result = elastic(
        *deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho),
        DX,
        DT,
        source_amplitudes_y=source_amplitudes,
        source_locations_y=source_locations,
        receiver_locations_y=receiver_locations,
        pml_width=[0, 20, 20, 20],
    )
    observed_data = result[-2]  # y amplitude receiver signals

    # Re-scale to a reasonable scale
    observed_data = observed_data * 1e7

    # Add noise to the observed data
    noise_dist = (
        dist.Normal(observed_data, observation_noise_scale).to_event(3).expand([N])
    )
    noisy_obs = pyro.sample(
        "receiver_observations",
        noise_dist,
        obs=receiver_observations,
    )
    return noisy_obs


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the different subsurface conditions for "nominal" and "failure"
    with torch.no_grad():
        source_amplitudes = (
            (deepwave.wavelets.ricker(FREQ, NT, DT, PEAK_TIME))
            .repeat(N_SHOTS, N_SOURCES_PER_SHOT, 1)
            .to(device)
        )

        profile_background = torch.zeros(NY_COARSE, NX_COARSE, device=device)

        # Nominal has a layer of higher vp, vs, and rho in the middle
        profile_nominal = profile_background.clone()
        profile_nominal[3:6, 1:9] = 1.0

        # Failure has a break in the middle of the layer
        profile_failure = profile_background.clone()
        profile_failure[3:6, 0:4] = 1.0
        profile_failure[4:7, 6:10] = 1.0

        # Generate the data for the nominal and failure cases
        N_nominal = 25
        N_failure = 2
        observation_noise_scale = 1e-2

        nominal_model = pyro.poutine.condition(
            seismic_model, data={"profile": profile_nominal}
        )
        nominal_observations = nominal_model(
            N=N_nominal, observation_noise_scale=observation_noise_scale
        )

        failure_model = pyro.poutine.condition(
            seismic_model, data={"profile": profile_failure}
        )
        failure_observations = failure_model(
            N=N_failure, observation_noise_scale=observation_noise_scale
        )

    sns.set(style="white", context="paper", color_codes=True)

    # Plot the profiles
    fig, axs = plt.subplots(2, 1, figsize=(4, 8))
    axs[0].imshow(profile_nominal.cpu().numpy(), cmap="Greys", alpha=0.5)
    axs[0].set_ylabel("Nominal")
    axs[1].imshow(profile_failure.cpu().numpy(), cmap="Greys", alpha=0.5)
    axs[1].set_ylabel("Anomaly")
    plt.tight_layout()
    plt.savefig("paper_plots/swi/swi_profiles.png", dpi=1000)
    plt.close()

    # Plot the observations
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(source_amplitudes[0, 0, :].cpu().numpy(), "k--", label="Source")
    ax.plot([], "b-", label="Nominal ($\\times 10^7$))")
    ax.plot([], "r-", label="Anomaly ($\\times 10^7$)")
    ax.legend()
    ax.plot(nominal_observations[:, 2, 4, :].cpu().numpy().T, "b-", linewidth=0.5)
    ax.plot(failure_observations[:, 2, 4, :].cpu().numpy().T, "r-", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("paper_plots/swi/swi_observations.png", dpi=1000)
    plt.close()
