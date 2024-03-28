"""Experiments using the learned models for classification."""


import os

import torch
import zuko
import pyro
from tqdm import tqdm

import pandas as pd
import bayes_air.utils.dataloader as ba_dataloader
from bayes_air.network import NetworkState
from bayes_air.model import air_traffic_network_model
from bayes_air.schedule import parse_schedule
from scripts.utils import ConditionalGaussianMixture, anomaly_classifier_metrics


def load_model(
    model_path,
    device,
    n_calibration_permutations,
    n_latent_variables,
    gmm=False,
    wasserstein=False,
    num_components=1,
):
    # Initialize the models
    if wasserstein:
        failure_guide = zuko.flows.CNF(
            features=n_latent_variables,
            context=n_calibration_permutations,
            hidden_features=(64, 64),
        ).to(device)
    elif gmm:
        failure_guide = ConditionalGaussianMixture(
            n_context=n_calibration_permutations, n_features=2
        ).to(device)
    else:
        failure_guide = zuko.flows.NSF(
            features=n_latent_variables,
            context=n_calibration_permutations,
            hidden_features=(64, 64),
        ).to(device)

    checkpoint = torch.load(model_path)
    failure_guide.load_state_dict(checkpoint["failure_guide"])
    mixture_label = checkpoint["mixture_label"].to(device)

    return failure_guide, mixture_label


@torch.no_grad
def make_eval_set(n, device):
    top_n = 4
    n_nominal = 9
    n_failure = 2
    include_cancellations = False
    per_point = False

    # Load the data
    df = pd.read_pickle("data/wn_data_clean_mst.pkl")
    df = ba_dataloader.top_N_df(df, top_n)
    nominal_df, disrupted_df = ba_dataloader.split_nominal_disrupted_data(df)
    nominal_dfs = ba_dataloader.split_by_date(nominal_df)
    disrupted_dfs = ba_dataloader.split_by_date(disrupted_df)

    # Get just the set of data we want to study
    nominal = nominal_dfs[-n_nominal:]
    nominal_eval = nominal_dfs[:n_nominal]

    if not per_point:
        failure = disrupted_dfs[: 2 * n_failure : 2]
        n_failure_eval = n_failure
        failure_eval = disrupted_dfs[1 : 1 + n_failure : 2]
    else:
        failure = disrupted_dfs[:n_failure]
        n_failure_eval = 1
        failure_eval = disrupted_dfs[:n_failure_eval]
        n_calibration_permutations = n_failure

    # Filter out cancellations if we're not using them
    if not include_cancellations:
        nominal = [df[~df["cancelled"]] for df in nominal]
        nominal_eval = [df[~df["cancelled"]] for df in nominal_eval]
        failure = [df[~df["cancelled"]] for df in failure]
        failure_eval = [df[~df["cancelled"]] for df in failure_eval]

    # Convert each day into a schedule
    nominal_states = []
    nominal_eval_states = []
    failure_states = []
    failure_eval_states = []

    for day_df in nominal:
        flights, airports = parse_schedule(day_df, device=device)

        state = NetworkState(
            airports={airport.code: airport for airport in airports},
            pending_flights=flights,
        )
        nominal_states.append(state)

    for day_df in nominal_eval:
        flights, airports = parse_schedule(day_df, device=device)

        state = NetworkState(
            airports={airport.code: airport for airport in airports},
            pending_flights=flights,
        )
        nominal_eval_states.append(state)

    for day_df in failure:
        flights, airports = parse_schedule(day_df, device=device)

        state = NetworkState(
            airports={airport.code: airport for airport in airports},
            pending_flights=flights,
        )
        failure_states.append(state)

    for day_df in failure_eval:
        flights, airports = parse_schedule(day_df, device=device)

        state = NetworkState(
            airports={airport.code: airport for airport in airports},
            pending_flights=flights,
        )
        failure_eval_states.append(state)

    return nominal_eval_states, failure_eval_states


def evaluate_model(
    model_name, gmm=False, wasserstein=False, n_calibration_permutations=5, n_eval=100
):
    device = torch.device("cpu")
    # Load eval data
    nominal_states_eval, failure_states_eval = make_eval_set(n_eval, device)
    n_nominal_eval = len(nominal_states_eval)
    n_failure_eval = len(failure_states_eval)

    dt = 0.2
    include_cancellations = False
    airport_codes = list(nominal_states_eval[0].airports.keys())
    n_airports = len(airport_codes)
    assert n_airports == 4

    model_path = os.path.join(
        "checkpoints", "wn", model_name, "failure_checkpoint_149.pt"
    )
    n_latent_variables = n_airports + n_airports + n_airports**2
    failure_guide, mixture_label = load_model(
        model_path,
        device,
        n_calibration_permutations,
        n_latent_variables,
        gmm,
        wasserstein,
    )

    # Make a score function for evaluating
    def map_to_sample_sites(sample):
        """Map vectorized samples to sample sites in the probabilistic model."""
        # Handle batched samples
        single_sample = len(sample.shape) == 1
        if single_sample:
            sample = sample.unsqueeze(0)

        assert sample.shape[-1] == n_latent_variables

        # Reshape & reparameterize the sample to satisfy positivity constraints
        airport_turnaround_times = torch.exp(sample[:, :n_airports])
        airport_service_times = torch.exp(sample[:, n_airports : 2 * n_airports])
        if include_cancellations:
            log_airport_initial_available_aircraft = sample[
                :, 2 * n_airports : 3 * n_airports
            ]
            log_airport_base_cancel_prob = sample[:, 3 * n_airports : 4 * n_airports]
            travel_times = torch.exp(
                sample[:, 4 * n_airports :].reshape(-1, n_airports, n_airports)
            )
        else:
            travel_times = torch.exp(
                sample[:, 2 * n_airports :].reshape(-1, n_airports, n_airports)
            )

        # Map to sample sites in the model
        conditioning_dict = {}
        for i, code in enumerate(airport_codes):
            conditioning_dict[
                f"{code}_mean_turnaround_time"
            ] = airport_turnaround_times[:, i]
            conditioning_dict[f"{code}_mean_service_time"] = airport_service_times[:, i]
            if include_cancellations:
                conditioning_dict[
                    f"{code}_log_initial_available_aircraft"
                ] = log_airport_initial_available_aircraft[:, i]
                conditioning_dict[
                    f"{code}_base_cancel_logprob"
                ] = log_airport_base_cancel_prob[:, i]

        for i, origin in enumerate(airport_codes):
            for j, destination in enumerate(airport_codes):
                if origin != destination:
                    conditioning_dict[
                        f"travel_time_{origin}_{destination}"
                    ] = travel_times[:, i, j]

        # Remove the batch dimension if it wasn't there before
        if single_sample:
            conditioning_dict = {
                key: value.squeeze(0) for key, value in conditioning_dict.items()
            }

        return conditioning_dict

    # Make the objective and divergence closures

    def single_particle_elbo(guide_dist, states):
        posterior_sample, posterior_logprob = guide_dist.rsample_and_log_prob()

        conditioning_dict = map_to_sample_sites(posterior_sample)

        model_trace = pyro.poutine.trace(
            pyro.poutine.condition(air_traffic_network_model, data=conditioning_dict)
        ).get_trace(
            states=states,
            delta_t=dt,
            device=device,
            include_cancellations=False,
        )
        model_logprob = model_trace.log_prob_sum()

        return model_logprob - posterior_logprob

    # Also make a closure for classifying anomalies
    def score_fn(nominal_guide_dist, failure_guide_dist, n, obs):
        scores = torch.zeros(n).to(device)

        n_samples = 10
        for i in tqdm(range(len(obs))):
            # nominal_elbo = torch.tensor(0.0).to(device)
            # for _ in range(n_samples):
            #     nominal_elbo += (
            #         single_particle_elbo(nominal_guide_dist, [obs[i]])
            #         / n_samples
            #     )

            failure_elbo = torch.tensor(0.0).to(device)
            for _ in range(n_samples):
                failure_elbo += (
                    single_particle_elbo(failure_guide_dist, [obs[i]]) / n_samples
                )

            # scores[i] = failure_elbo - nominal_elbo
            scores[i] = failure_elbo * 1e-1

        return scores

    # Evaluate the model
    with torch.no_grad():
        failure_scores = score_fn(
            failure_guide(torch.zeros_like(mixture_label)),
            failure_guide(mixture_label),
            failure_states_eval.shape[0],
            failure_states_eval,
        )
        nominal_scores = score_fn(
            failure_guide(torch.zeros_like(mixture_label)),
            failure_guide(mixture_label),
            nominal_states_eval.shape[0],
            nominal_states_eval,
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
            "model_name": "ours_calibrated__3/2024-03-27_07-20-04",
            "gmm": False,
            "wasserstein": False,
            "n_calibration_permutations": 5,
        },
        # {
        #     "model_name": "kl_regularized_kl",
        #     "gmm": False,
        #     "wasserstein": False,
        #     "n_calibration_permutations": 5,
        # },
        # {
        #     "model_name": "w2_regularized",
        #     "gmm": False,
        #     "wasserstein": True,
        #     "n_calibration_permutations": 5,
        # },
        # {
        #     "model_name": "ours_gmm_calibrated_",
        #     "gmm": True,
        #     "wasserstein": False,
        #     "n_calibration_permutations": 5,
        # },
    ]

    for eval_dict in to_eval:
        evaluate_model(**eval_dict)
