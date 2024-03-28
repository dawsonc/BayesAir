"""Experiments using the learned models for classification."""


import os

import torch
import zuko
import pyro

from scripts.swi.model import NX_COARSE, NY_COARSE, seismic_model
from scripts.utils import ConditionalGaussianMixture, anomaly_classifier_metrics


def load_model(
    model_path,
    device,
    n_calibration_permutations,
    gmm=False,
    wasserstein=False,
    num_components=1,
):
    # Load the model
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

    checkpoint = torch.load(model_path)
    failure_guide.load_state_dict(checkpoint["failure_guide"])
    mixture_label = checkpoint["mixture_label"].to(device)

    return failure_guide, mixture_label


@torch.no_grad
def make_eval_set(n, device):
    profile_background = torch.zeros(NY_COARSE, NX_COARSE, device=device)

    n_failure_eval = n

    observation_noise_scale = 1e-2

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

    return failure_observations_eval, nominal_observations_eval


def evaluate_model(
    model_name, gmm=False, wasserstein=False, n_calibration_permutations=5, n_eval=100
):
    device = torch.device("cpu")
    model_path = os.path.join(
        "checkpoints", "swi", model_name, "failure_checkpoint_499.pt"
    )
    failure_guide, mixture_label = load_model(
        model_path, device, n_calibration_permutations, gmm, wasserstein
    )

    # Load eval data
    failure_observations_eval, nominal_observations_eval = make_eval_set(n_eval, device)
    n_nominal_eval = nominal_observations_eval.shape[0]
    n_failure_eval = failure_observations_eval.shape[0]

    # Make a score function for evaluating
    def single_particle_elbo(guide_dist, n, obs):
        posterior_sample, posterior_logprob = guide_dist.rsample_and_log_prob()

        # Reshape the sample
        posterior_sample = posterior_sample.reshape(NY_COARSE, NX_COARSE)

        model_trace = pyro.poutine.trace(
            pyro.poutine.condition(seismic_model, data={"profile": posterior_sample})
        ).get_trace(N=n, receiver_observations=obs, device=obs.device)
        model_logprob = model_trace.log_prob_sum()

        return model_logprob - posterior_logprob

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

    # Evaluate the model
    with torch.no_grad():
        failure_scores = score_fn(
            failure_guide(torch.zeros_like(mixture_label)),
            failure_guide(mixture_label),
            failure_observations_eval.shape[0],
            failure_observations_eval,
        )
        nominal_scores = score_fn(
            failure_guide(torch.zeros_like(mixture_label)),
            failure_guide(mixture_label),
            nominal_observations_eval.shape[0],
            nominal_observations_eval,
        )

        labels = (
            torch.cat([torch.zeros(n_nominal_eval), torch.ones(n_failure_eval)])
            .to(device)
            .to(torch.int32)
        )
        scores = torch.cat([nominal_scores, failure_scores])
        aucroc, aucpr, precision, recall = anomaly_classifier_metrics(scores, labels)
        print("-------------------------------------")
        print(f"Model: {model_name}")
        print(
            f"AUCROC: {aucroc}, AUCPR: {aucpr}, Precision: {precision}, Recall: {recall}"
        )
        return aucroc, aucpr, precision, recall


if __name__ == "__main__":
    to_eval = [
        {
            "model_name": "ours_calibrated_",
            "gmm": False,
            "wasserstein": False,
            "n_calibration_permutations": 5,
        },
        {
            "model_name": "kl_regularized_kl",
            "gmm": False,
            "wasserstein": False,
            "n_calibration_permutations": 5,
        },
        {
            "model_name": "w2_regularized",
            "gmm": False,
            "wasserstein": True,
            "n_calibration_permutations": 5,
        },
        {
            "model_name": "ours_gmm_calibrated_",
            "gmm": True,
            "wasserstein": False,
            "n_calibration_permutations": 5,
        },
    ]

    for eval_dict in to_eval:
        evaluate_model(**eval_dict)
