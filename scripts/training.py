"""Implement the training loop for our method and baselines."""
import torch
from tqdm import tqdm

import wandb


def train(
    nominal_guide,
    n_nominal,
    nominal_observations,
    failure_guide,
    n_failure,
    failure_observations,
    n_failure_eval,
    failure_observations_eval,
    objective_fn,
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
    calibration_ub,
    calibration_lr,
    plot_every_n=10,
):
    """
    Compute the loss for the seismic waveform inversion problem.

    Args:
        nominal_guide: the nominal model. Should take no context
        n_nominal: Number of nominal observations.
        nominal_observations: Observed data for the nominal case.
        failure_guide: the failure model. Should take context of length
            calibration_num_permutations
        n_failure: Number of failure observations.
        failure_observations: Observed data for the failure case.
        n_failure_eval: Number of failure observations for evaluation.
        failure_observations_eval: Observed data for the failure case for evaluation.
        objective_fn: Function for computing the inference objective (e.g. likelihood or
            ELBO).
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
        calibration_ub: upper bound on calibration divergence
        calibration_lr: learning rate for calibration
        plot_every_n: number of steps between plotting
    """
    device = nominal_observations.device

    # Set up the optimizers
    nominal_optimizer = torch.optim.Adam(
        nominal_guide.parameters(), lr=lr, weight_decay=weight_decay
    )
    nominal_scheduler = torch.optim.lr_scheduler.StepLR(
        nominal_optimizer, step_size=lr_steps, gamma=lr_gamma
    )
    failure_optimizer = torch.optim.Adam(
        failure_guide.transform.parameters(),  # only transform in case base is a flow
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
        nominal_elbo = objective_fn(nominal_guide(), n_nominal, nominal_observations)
        loss_nominal = nominal_elbo  # ELBO is returned as a loss, already negative
        loss_nominal.backward()
        nominal_grad_norm = torch.nn.utils.clip_grad_norm_(
            nominal_guide.parameters(), grad_clip
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
                failure_elbo += objective_fn(
                    failure_guide(label),
                    failure_permutations[j].shape[0],
                    failure_observations[failure_permutations[j]],
                )

            failure_elbo = failure_elbo / calibration_num_permutations
            failure_loss += elbo_weight * failure_elbo
        else:
            # If not calibrating, just get the ELBO on the failure observations
            failure_elbo = objective_fn(
                failure_guide(torch.ones(calibration_num_permutations).to(device)),
                n_failure,
                failure_observations,
            )
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
            samples = failure_guide(random_labels).rsample((100,))
            kl_p_base = (
                failure_guide(random_labels).log_prob(samples)
                - failure_guide.base(random_labels).log_prob(samples)
            ).mean(dim=0)
            kl_err = (
                kl_p_base
                - calibration_ub
                * torch.norm(random_labels, dim=-1)
                / calibration_num_permutations
            )
            mean_calibration_error = torch.mean(kl_err**2)
            failure_loss += calibration_weight * mean_calibration_error
        elif regularize:
            if wasserstein_regularization:
                # Get the square norm of the flow
                ode_dim = failure_guide.transform.ode[0].in_features
                samples = torch.randn(100, ode_dim).to(device)
                mean_square_flow = torch.mean(
                    (failure_guide.transform.ode(samples) ** 2).sum(dim=-1)
                )
                failure_loss += regularization_weight * mean_square_flow
            else:
                # Get the KL divergence between the failure and nominal models
                regularization_kl = divergence_fn(
                    failure_guide(torch.ones(calibration_num_permutations).to(device)),
                    nominal_guide(),
                )
                failure_loss += regularization_weight * regularization_kl

        # Also make sure the failure distribution can represent the nominal
        # distribution with a zero label
        failure_nominal_elbo = objective_fn(
            failure_guide(torch.zeros(calibration_num_permutations).to(device)),
            n_nominal,
            nominal_observations,
        )
        failure_loss += elbo_weight * failure_nominal_elbo

        # Optimimze the failure model
        failure_loss.backward()
        failure_grad_norm = torch.nn.utils.clip_grad_norm_(
            failure_guide.parameters(), grad_clip
        )
        failure_optimizer.step()
        failure_scheduler.step()

        # Record progress
        if i % plot_every_n == 0 or i == num_steps - 1:
            nominal_label = torch.zeros(calibration_num_permutations).to(device)
            failure_label = torch.ones(calibration_num_permutations).to(device)
            calibrated_objective = None
            if calibrate:
                failure_label = torch.zeros(
                    calibration_num_permutations, requires_grad=True
                )
                failure_label_optimizer = torch.optim.Adam(
                    [failure_label], lr=calibration_lr
                )
                for j in range(1000):
                    failure_label_optimizer.zero_grad()
                    failure_label_loss = objective_fn(
                        failure_guide(failure_label),
                        n_failure,
                        failure_observations,
                    )
                    failure_label_loss.backward()
                    failure_label_optimizer.step()

                calibrated_objective = failure_label_loss.detach().cpu().item()

            with torch.no_grad():
                # Evaluate the failure model
                failure_objective_eval = objective_fn(
                    failure_guide(failure_label),
                    n_failure_eval,
                    failure_observations_eval,
                )

                plot_posterior(
                    nominal_guide(),
                    failure_guide(failure_label),
                    labels=["Nominal", "Failure (calibrated)"],
                    save_file_name=None,
                    save_wandb=True,
                )

                if calibrate:
                    plot_posterior_grid(
                        nominal_guide,
                        failure_guide,
                        nominal_label,
                        save_file_name=None,
                        save_wandb=True,
                    )

            torch.save(
                nominal_guide.state_dict(),
                f"checkpoints/{name}/nominal_checkpoint_{i}.pt",
            )
            torch.save(
                {
                    "failure_guide": failure_guide.state_dict(),
                    "failure_label": failure_label,
                },
                f"checkpoints/{name}/failure_checkpoint_{i}.pt",
            )

        wandb.log(
            {
                "Nominal/ELBO": nominal_elbo.detach().cpu().item(),
                "Nominal/loss": loss_nominal.detach().cpu().item(),
                "Nominal/gradient norm": nominal_grad_norm.detach().cpu().item(),
                "Failure/ELBO": failure_elbo.detach().cpu().item(),
                "Failure/ELBO (eval)": failure_objective_eval.detach().cpu().item(),
                "Failure/loss": failure_loss.detach().cpu().item(),
                "Failure/gradient norm": failure_grad_norm.detach().cpu().item(),
            }
            | (
                {
                    "Failure/mean calibration error": mean_calibration_error.detach()
                    .cpu()
                    .item(),
                }
                if calibrate
                else {}
            )
            | (
                {"Failure/ELBO (calibrated)": calibrated_objective}
                if calibrated_objective is not None
                else {}
            )
            | (
                {
                    "Failure/regularization KL": regularization_kl.detach()
                    .cpu()
                    .item(),
                }
                if regularize and not wasserstein_regularization
                else {}
            )
            | (
                {
                    "Failure/mean square flow": mean_square_flow.detach().cpu().item(),
                }
                if regularize and wasserstein_regularization
                else {}
            )
        )

    return nominal_guide, failure_guide, failure_label
