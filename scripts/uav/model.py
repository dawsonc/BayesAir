"""Define a simplified model for the UAV attitude dynamics."""
import pyro
import pyro.distributions as dist
import torch


def model(
    initial_states,
    commands,
    observed_next_states=None,
    observed_pqrs=None,
    dt=0.25,
    device=None,
):
    """Define a simplified model for the UAV attitude dynamics.

    Args:
        initial_states: N x 3 tensor of initial states (roll, pitch, yaw) for the UAV
        commands: N x 3 tensor of commands (desired roll, pitch, yaw)
        observed_next_states: N x 3 tensor of observed next states (roll, pitch, yaw)
        observed_pqrs: N x 3 tensor of observed angular velocities (p, q, r)
        dt: Time step between states
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check consistency of batch and time dimensions
    N = initial_states.shape[0]
    assert initial_states.shape == (N, 3)
    assert commands.shape == (N, 3)
    assert observed_next_states is None or observed_next_states.shape == (N, 3)
    assert observed_pqrs is None or observed_pqrs.shape == (N, 3)

    # Use attitude dyamics with state x = [phi, theta, psi] (roll, pitch, yaw)
    # dx/dt = J^-1 * (Ax + Ke + d + eta)
    # where J^-1 is the kinematics, A is the state-to-state transfer matrix, K is the
    # error-to-state transfer matrix, e is the error, d is a constant bias, and eta is
    # Gaussian noise.

    # Sample the matrices from the prior
    A = pyro.sample(
        "A",
        dist.Normal(torch.zeros(3, 3, device=device), torch.ones(3, 3, device=device)),
    )
    K = pyro.sample(
        "K",
        dist.Normal(torch.zeros(3, 3, device=device), torch.ones(3, 3, device=device)),
    )
    d = pyro.sample(
        "d", dist.Normal(torch.zeros(3, device=device), torch.ones(3, device=device))
    ).reshape(3, 1)
    log_noise_strength = pyro.sample(
        "log_noise_strength", dist.Normal(torch.tensor(-2.0, device=device), 1.0)
    )
    noise_strength = torch.exp(log_noise_strength)

    # Compute the next state for each example in the data
    states = initial_states.reshape(-1, 3, 1)
    commands = commands.reshape(-1, 3, 1)
    errors = commands - states

    # Compute the mean velocity based on the system matrices
    pqr_mean = A @ states + K @ errors + d
    noise_dist = (
        dist.Normal(pqr_mean, noise_strength * torch.ones_like(pqr_mean))
        .to_event(2)
        .expand([N])
    )
    pqrs = pyro.sample(
        "pqrs",
        noise_dist,
        obs=observed_pqrs.reshape(-1, 3, 1) if observed_pqrs is not None else None,
    )
    action_noise = pqrs - pqr_mean

    # Construct the kinematic matrix
    roll, pitch = states[:, 0, 0], states[:, 1, 0]
    Jinv = torch.zeros(N, 3, 3, device=device)
    Jinv[:, 0, 0] = 1.0
    Jinv[:, 0, 1] = torch.tan(pitch) * torch.sin(roll)
    Jinv[:, 0, 2] = torch.tan(roll) * torch.cos(pitch)
    Jinv[:, 1, 1] = torch.cos(roll)
    Jinv[:, 1, 2] = -torch.sin(roll)
    Jinv[:, 2, 1] = torch.sin(roll) / torch.cos(pitch)
    Jinv[:, 2, 2] = torch.cos(roll) / torch.cos(pitch)

    # Integrate the change in state
    next_states = states + dt * Jinv @ pqrs
    noise_dist = (
        dist.Normal(next_states, noise_strength * torch.ones_like(next_states))
        .to_event(2)
        .expand([N])
    )
    observed_states = pyro.sample(
        "states",
        noise_dist,
        obs=observed_next_states.reshape(next_states.shape)
        if observed_next_states is not None
        else None,
    )
    state_observation_noise = observed_states - next_states

    return next_states, pqrs, state_observation_noise, action_noise
