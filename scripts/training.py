"""Implement the training loop for our method and baselines."""
import torch
from tqdm import tqdm

import wandb
from scripts.utils import cross_entropy, f_score, simple_mmd


def train(
    n_nominal,
    nominal_observations,
    failure_guide,
    n_failure,
    failure_observations,
    n_failure_eval,
    failure_observations_eval,
    failure_posterior_samples_eval,
    nominal_posterior_samples_eval,
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
    calibration_substeps,
    calibration_steps=200,
    plot_every_n=10,
    device=None,
    exclude_nominal=False,
):
    """
    Compute the loss for the seismic waveform inversion problem.

    Args:
        n_nominal: Number of nominal observations.
        nominal_observations: Observed data for the nominal case.
        failure_guide: the failure model. Should take context of length
            calibration_num_permutations
        n_failure: Number of failure observations.
        failure_observations: Observed data for the failure case.
        n_failure_eval: Number of failure observations for evaluation.
        failure_observations_eval: Observed data for the failure case for evaluation.
        failure_posterior_samples_eval: Samples from the failure posterior for
            evaluation.
        nominal_posterior_samples_eval: Samples from the nominal posterior for
            evaluation.
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
        calibration_substeps: number of calibration steps to take per training step
        calibration_steps: number of calibration steps to take prior to evaluation
        plot_every_n: number of steps between plotting
        device: device to use for training
        exclude_nominal: if True, exclude the nominal data from the failure model
            training
    """
    if device is None:
        device = nominal_observations.device

    # Set up the optimizers
    failure_optimizer = torch.optim.Adam(
        failure_guide.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    failure_scheduler = torch.optim.lr_scheduler.StepLR(
        failure_optimizer, step_size=lr_steps, gamma=lr_gamma
    )

    # If the calibration lr is zero, freeze the mixture label
    if calibration_lr == 0:
        mixture_label = (
            torch.ones(calibration_num_permutations, requires_grad=False, device=device)
            / calibration_num_permutations
        )
    else:
        mixture_label = torch.tensor(
            [1 / calibration_num_permutations] * calibration_num_permutations,
            requires_grad=True,
            device=device,
        )
    mixture_label_optimizer = torch.optim.Adam([mixture_label], lr=calibration_lr)

    # Create the permutations for training the calibrated model
    failure_permutations = []
    # Make sure the first two permutations cover the data
    assert calibration_num_permutations >= 2, "Need at least two permutations"
    first_permutation = torch.randperm(n_failure)
    failure_permutations.append(first_permutation[: n_failure // 2])
    failure_permutations.append(first_permutation[-n_failure // 2 :])
    for i in range(2, calibration_num_permutations):
        failure_permutations.append(torch.randperm(n_failure)[: n_failure // 2])

    # Add the permutations to a wandb table
    if calibrate:
        columns = [f"Permutation member {i}" for i in range(n_failure // 2)]
        data = [
            failure_permutation.cpu().tolist()
            for failure_permutation in failure_permutations
        ]
        table = wandb.Table(data=data, columns=columns)
        wandb.log({"Failure data permutations": table}, commit=False)

    # Train the model
    pbar = tqdm(range(num_steps))
    for i in pbar:
        # Train the mixture label
        if calibrate:
            for _ in range(calibration_substeps):
                mixture_label_optimizer.zero_grad()
                mixture_label_loss = objective_fn(
                    failure_guide(mixture_label), n_failure, failure_observations
                )
                mixture_label_loss.backward()
                mixture_grad_norm = torch.nn.utils.clip_grad_norm_(
                    mixture_label, grad_clip
                )
                mixture_label_optimizer.step()

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
                if isinstance(failure_observations, list):
                    subsample = [
                        failure_observations[k] for k in failure_permutations[j]
                    ]
                else:
                    subsample = failure_observations[failure_permutations[j]]

                failure_elbo += objective_fn(
                    failure_guide(label),
                    failure_permutations[j].shape[0],
                    subsample,
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

        # The second part of the failure loss is regularization/calibration
        # There are three cases:
        #   - calibrating (ours): Get the performance of the calibrated label on the
        #     training set
        #   - not calibrating, regularizing with KL: get the KL divergence between the
        #     failure model and the nominal model
        #   - not calibrating, regularizing with Wasserstein: penalize the square norm
        #     of the failure model's ode flow (only works for CNF)
        if calibrate:
            mixture_label_loss = objective_fn(
                failure_guide(mixture_label), n_failure, failure_observations
            )
            failure_loss += calibration_weight * mixture_label_loss
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
                    failure_guide(torch.zeros(calibration_num_permutations).to(device)),
                )
                failure_loss += regularization_weight * regularization_kl

        # Also make sure the failure distribution can represent the nominal
        # distribution with a zero label
        failure_nominal_elbo = objective_fn(
            failure_guide(torch.zeros(calibration_num_permutations).to(device)),
            n_nominal,
            nominal_observations,
        )
        if not exclude_nominal:
            failure_loss += elbo_weight * failure_nominal_elbo

        # Optimimze the failure model
        failure_loss.backward()
        failure_grad_norm = torch.nn.utils.clip_grad_norm_(
            failure_guide.parameters(), grad_clip
        )
        failure_optimizer.step()
        failure_scheduler.step()

        # Evaluate the failure model
        with torch.no_grad():
            failure_objective_eval = objective_fn(
                failure_guide(mixture_label),
                n_failure_eval,
                failure_observations_eval,
            )

        # Record progress
        if i % plot_every_n == 0 or i == num_steps - 1:
            if calibrate:
                for _ in range(calibration_steps):
                    mixture_label_optimizer.zero_grad()
                    mixture_label_loss = objective_fn(
                        failure_guide(mixture_label), n_failure, failure_observations
                    )
                    mixture_label_loss.backward()
                    mixture_grad_norm = torch.nn.utils.clip_grad_norm_(
                        mixture_label, grad_clip
                    )
                    mixture_label_optimizer.step()

            with torch.no_grad():
                plot_posterior(
                    failure_guide(torch.zeros(calibration_num_permutations).to(device)),
                    failure_guide(mixture_label)
                    if calibrate
                    else failure_guide(
                        torch.ones(calibration_num_permutations).to(device)
                    ),
                    labels=["Nominal", "Failure (calibrated)"],
                    save_file_name=None,
                    save_wandb=True,
                )

                plot_posterior_grid(
                    failure_guide,
                    torch.zeros(calibration_num_permutations).to(device),
                    save_file_name=None,
                    save_wandb=True,
                )

            torch.save(
                {
                    "failure_guide": failure_guide.state_dict(),
                    "mixture_label": mixture_label,
                    "failure_permutations": failure_permutations,
                },
                f"checkpoints/{name}/failure_checkpoint_{i}.pt",
            )

        # Compare the failure model to the eval data via a couple of metrics
        with torch.no_grad():
            nominal_dist = failure_guide(
                torch.zeros(calibration_num_permutations).to(device)
            )
            if calibrate:
                failure_dist = failure_guide(mixture_label)
            else:
                failure_dist = failure_guide(
                    torch.ones(calibration_num_permutations).to(device)
                )

            if failure_posterior_samples_eval is not None:
                n_eval = failure_posterior_samples_eval.shape[0]
                p_samples_eval = failure_dist.sample((n_eval,))
                mmd = simple_mmd(p_samples_eval, failure_posterior_samples_eval)
                ce = cross_entropy(failure_posterior_samples_eval, failure_dist)

                if nominal_posterior_samples_eval is not None:
                    f_score_eval = f_score(
                        nominal_posterior_samples_eval,
                        failure_posterior_samples_eval,
                        nominal_dist,
                        failure_dist,
                    )

        wandb.log(
            {
                "Failure/ELBO": failure_elbo.detach().cpu().item(),
                "Failure/ELBO (eval)": failure_objective_eval.detach().cpu().item(),
                "Failure/MMD (eval)": mmd
                if failure_posterior_samples_eval is not None
                else None,
                "Failure/F score (eval)": f_score_eval
                if failure_posterior_samples_eval is not None
                and nominal_posterior_samples_eval is not None
                else None,
                "Failure/CE (eval)": ce
                if failure_posterior_samples_eval is not None
                else None,
                "Failure/loss": failure_loss.detach().cpu().item(),
                "Failure/gradient norm": failure_grad_norm.detach().cpu().item(),
            }
            # | (
            #     {
            #         "Failure/mean calibration error": mean_calibration_error.detach()
            #         .cpu()
            #         .item(),
            #     }
            #     if calibrate
            #     else {}
            # )
            | (
                {
                    "Failure/ELBO (calibrated)": mixture_label_loss.detach()
                    .cpu()
                    .item(),
                    "Failure/mixture grad norm": mixture_grad_norm.detach()
                    .cpu()
                    .item(),
                }
                if calibrate
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

    # Save the model to wandb
    wandb.save(f"checkpoints/{name}/failure_checkpoint_{num_steps - 1}.pt")

    return failure_guide, mixture_label
