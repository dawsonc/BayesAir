"""Implement the training loop for our method and baselines."""
import torch
from tqdm import tqdm

import wandb


def train(
    nominal_model,
    n_nominal,
    nominal_observations,
    failure_model,
    n_failure,
    failure_observations,
    elbo_fn,
    divergence_fn,
    plot_posterior,
    plot_posterior_grid,
    name,
    calibrate,
    regularize,
    num_steps,
    lr,
    lr_gamma,
    lr_steps,
    grad_clip,
    weight_decay,
    num_calibration_points,
    calibration_weight,
    regularization_weight,
    elbo_weight,
    wasserstein_regularization,
    calibration_num_permutations,
):
    """
    Compute the loss for the seismic waveform inversion problem.

    Args:
        nominal_model: the nominal model. Should take no context
        n_nominal: Number of nominal observations.
        nominal_observations: Observed data for the nominal case.
        failure_model: the failure model. Should take context of length
            calibration_num_permutations
        n_failure: Number of failure observations.
        failure_observations: Observed data for the failure case.
        elbo_fn: Function for computing the ELBO.
        divergence_fn: Function for computing the divergence.
        plot_posterior: Function for plotting the posterior.
        plot_posterior_grid: Function for plotting the posterior grid.
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
        num_calibration_points: number of points for divergence calibration
        calibration_weight: weight applied to calibration loss
        regularization_weight: weight applied to nominal divergence loss
        elbo_weight: weight applied to ELBO loss
        wasserstein_regularization: if True, use Wasserstein for regularization rather
            than KL. Only compatible with CNF failure flows.
        calibration_num_permutations: number of permutations for calibration
    """
    device = nominal_observations.device

    # Set up the optimizers
    nominal_optimizer = torch.optim.Adam(
        nominal_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    nominal_scheduler = torch.optim.lr_scheduler.StepLR(
        nominal_optimizer, step_size=lr_steps, gamma=lr_gamma
    )
    failure_optimizer = torch.optim.Adam(
        failure_model.transform.parameters(),  # only transform in case base is a flow
        lr=lr,
        weight_decay=weight_decay,
    )
    failure_scheduler = torch.optim.lr_scheduler.StepLR(
        failure_optimizer, step_size=lr_steps, gamma=lr_gamma
    )

    # Create the permutations for training the calibrated model
    failure_permutations = []
    for i in range(calibration_num_permutations):
        failure_permutations.append(torch.randperm(n_failure)[: n_failure // 2])

    # Train the model
    pbar = tqdm(range(num_steps))
    for i in pbar:
        # Optimize the nominal model
        nominal_optimizer.zero_grad()
        nominal_elbo = elbo_fn(nominal_model(), n_nominal, nominal_observations)[0]
        loss_nominal = nominal_elbo  # component[0] is ELBO loss, already negative
        loss_nominal.backward()
        nominal_grad_norm = torch.nn.utils.clip_grad_norm_(
            nominal_model.parameters(), grad_clip
        )
        nominal_optimizer.step()
        nominal_scheduler.step()

        # Optimize the failure model
        failure_optimizer.zero_grad()
        failure_loss = torch.tensor(0.0).to(device)

        # The first part of the failure loss is the ELBO on each of the failure
        # permutations
        if calibrate:
            failure_elbo = torch.tensor(0.0).to(device)
            for j in range(calibration_num_permutations):
                label = torch.zeros(calibration_num_permutations).to(device)
                label[j] = 1.0
                failure_elbo += elbo_fn(
                    failure_model(label),
                    failure_permutations[j].shape[0],
                    failure_observations[failure_permutations[j]],
                )[0]

            failure_elbo = failure_elbo / calibration_num_permutations
            failure_loss += elbo_weight * failure_elbo
        else:
            # If not calibrating, just get the ELBO on the failure observations
            failure_elbo = elbo_fn(
                failure_model(torch.ones(calibration_num_permutations).to(device)),
                n_failure,
                failure_observations,
            )[0]
            failure_loss += elbo_weight * failure_elbo

        # The second part of the failure loss is divergence based.
        # There are three cases:
        #   - calibrating (ours): get the KL divergence between the failure model and
        #     its base at some randomly sampled labels, then make the KL divergence
        #     roughly match the norm of those labels
        #   - not calibrating, regularizing with KL: get the KL divergence between the
        #     failure model and the nominal model
        #   - not calibrating, regularizing with Wasserstein: penalize the square norm
        #     of the failure model's ode flow (only works for CNF)
        if calibrate:
            random_labels = torch.rand(
                num_calibration_points, calibration_num_permutations
            )
            samples = failure_model(random_labels).rsample((100,))
            kl_p_base = (
                failure_model(random_labels).log_prob(samples)
                - failure_model.base(random_labels).log_prob(samples)
            ).mean(dim=0)
            kl_err = (
                kl_p_base
                - 100 * torch.norm(random_labels, dim=-1) / calibration_num_permutations
            )
            mean_calibration_error = torch.mean(kl_err**2)
            failure_loss += calibration_weight * mean_calibration_error
        elif regularize:
            if wasserstein_regularization:
                # Get the square norm of the flow
                ode_dim = failure_model.transform.ode[0].in_features
                samples = torch.randn(100, ode_dim).to(device)
                mean_square_flow = torch.mean(
                    (failure_model.transform.ode(samples) ** 2).sum(dim=-1)
                )
                failure_loss += regularization_weight * mean_square_flow
            else:
                # Get the KL divergence between the failure and nominal models
                regularization_kl = divergence_fn(
                    failure_model(torch.ones(calibration_num_permutations).to(device)),
                    nominal_model(),
                )
                failure_loss += regularization_weight * regularization_kl

        # Also make sure the failure distribution can represent the nominal
        # distribution with a zero label
        failure_nominal_elbo = elbo_fn(
            lambda: failure_model(torch.zeros(calibration_num_permutations).to(device)),
            n_nominal,
            nominal_observations,
        )[0]
        failure_loss += elbo_weight * failure_nominal_elbo

        # Optimimze the failure model
        failure_loss.backward()
        failure_grad_norm = torch.nn.utils.clip_grad_norm_(
            failure_model.parameters(), grad_clip
        )
        failure_optimizer.step()
        failure_scheduler.step()

        # Record progress
        if i % 10 == 0 or i == num_steps - 1:
            nominal_label = torch.zeros(calibration_num_permutations).to(device)
            failure_label = torch.ones(calibration_num_permutations).to(device)
            if calibrate:
                failure_label = torch.zeros(
                    calibration_num_permutations, requires_grad=True
                )
                failure_label_optimizer = torch.optim.Adam([failure_label], lr=1e-3)
                for j in range(1000):
                    failure_label_optimizer.zero_grad()
                    failure_label_loss = elbo_fn(
                        failure_model(failure_label),
                        n_failure,
                        failure_observations,
                    )[0]
                    failure_label_loss.backward()
                    failure_label_optimizer.step()

            plot_posterior(
                nominal_model,
                failure_model,
                nominal_label,
                failure_label,
                save_file_name=name + "_posterior.png",
            )
            plot_posterior_grid(
                nominal_model,
                failure_model,
                nominal_label,
                save_file_name=name + "_posterior_grid.png",
            )

            torch.save(
                nominal_model.state_dict(),
                f"checkpoints/{name}/nominal_checkpoint_{i}.pt",
            )
            torch.save(
                {
                    "failure_model": failure_model.state_dict(),
                    "failure_label": failure_label,
                },
                f"checkpoints/{name}/failure_checkpoint_{i}.pt",
            )

        wandb.log(
            {
                "Nominal/ELBO": nominal_elbo.detatch().cpu().item(),
                "Nominal/loss": loss_nominal.detatch().cpu().item(),
                "Nominal/gradient norm": nominal_grad_norm.detatch().cpu().item(),
                "Failure/ELBO": failure_elbo.detatch().cpu().item(),
                "Failure/loss": failure_loss.detatch().cpu().item(),
                "Failure/gradient norm": failure_grad_norm.detatch().cpu().item(),
            }
            | (
                {
                    "Failure/mean calibration error": mean_calibration_error.detatch()
                    .cpu()
                    .item(),
                }
                if calibrate
                else {}
            )
            | (
                {
                    "Failure/regularization KL": regularization_kl.detatch()
                    .cpu()
                    .item(),
                }
                if regularize and not wasserstein_regularization
                else {}
            )
            | (
                {
                    "Failure/mean square flow": mean_square_flow.detatch().cpu().item(),
                }
                if regularize and wasserstein_regularization
                else {}
            )
        )
        pbar.set_description(
            f"Epoch {i} failure loss: {failure_loss.detatch().cpu().item():.3f}"
        )

    return nominal_model, failure_model, failure_label
