"""Run the simulation for a simple two-airport network."""
from copy import deepcopy

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import pyro
import seaborn as sns
import torch
import tqdm

from bayes_air.model import air_traffic_network_model
from bayes_air.network import NetworkState
from bayes_air.schedule import parse_schedule


def main():
    pyro.enable_validation(True)
    pyro.set_rng_seed(1)

    # Define a simple schedule that sends the aircraft from airport 0 to airport 1 and back
    # Time is defined in hours from the start of the simulation
    states = []
    for _ in range(10):
        schedule = pd.DataFrame(
            {
                "flight_number": ["F1", "F1"],
                "origin_airport": ["A1", "A2"],
                "destination_airport": ["A2", "A1"],
                "scheduled_departure_time": [0.0, 2.0],
                "scheduled_arrival_time": [1.0, 3.5],
                "actual_departure_time": [
                    0.0 + torch.rand(()) * 0.3,
                    2.0 + torch.rand(()) * 0.3,
                ],
                "actual_arrival_time": [
                    1.0 + torch.rand(()) * 0.3,
                    3.5 + torch.rand(()) * 0.3,
                ],
            }
        )
        flights, airports = parse_schedule(schedule)
        airports[0].available_aircraft.append(torch.tensor(0.0))
        airports[0].available_crew.append(torch.tensor(0.0))
        state = NetworkState(
            airports={airport.code: airport for airport in airports},
            pending_flights=flights,
        )
        states.append(state)

    hrs = 24.0
    dt = 0.1

    model = air_traffic_network_model

    model_graph = pyro.render_model(
        model,
        model_args=(deepcopy(states), hrs, dt),
        render_params=False,
        render_distributions=False,
    )
    model_graph.render("tmp/two_airport_network", cleanup=True)

    # # Print dependencies
    # print("Dependencies:")
    # pprint.pprint(
    #     pyro.infer.inspect.get_dependencies(
    #         model, (deepcopy(states), hrs, dt)
    #     )
    # )

    # # Set up MCMC inference
    # num_samples = 50
    # warmup_steps = 50
    # num_chains = 1
    # nuts_kernel = pyro.infer.NUTS(
    #     model,
    #     max_tree_depth=5,
    #     jit_compile=False,
    #     step_size=1.0,
    #     # adapt_step_size=False,
    # )
    # mcmc = pyro.infer.MCMC(
    #     nuts_kernel,
    #     num_samples=num_samples,
    #     warmup_steps=warmup_steps,
    #     num_chains=num_chains,
    # )
    # mcmc.run(deepcopy(states), hrs, dt)
    # mcmc.summary(prob=0.9)

    auto_guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
    adam = pyro.optim.Adam({"lr": 0.02})  # Consider decreasing learning rate.
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

    losses = []
    pbar = tqdm.tqdm(range(300))
    for step in pbar:
        loss = svi.step(states, hrs, dt)
        losses.append(loss)
        pbar.set_description(f"ELBO loss: {loss:.2f}")

    # Sample nominal travel time estimates from the posterior
    n_samples = 800
    with pyro.plate("samples", n_samples, dim=-1):
        posterior_samples = auto_guide(states, hrs, dt)

    # Because this is a simple problem, we know the exact distribution of travel times,
    # so we can compare the posterior to the true distribution
    true_travel_times_A1_A2 = (
        1 + torch.rand((n_samples,)) * 0.3 - torch.rand((n_samples,)) * 0.3
    )
    true_travel_times_A2_A1 = (
        1.5 + torch.rand((n_samples,)) * 0.3 - torch.rand((n_samples,)) * 0.3
    )

    # Plot training curve and travel time posterior
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(losses)
    axs[0].set_xlabel("SVI step")
    axs[0].set_ylabel("ELBO loss")

    sns.kdeplot(
        x=true_travel_times_A1_A2.detach().cpu().numpy(),
        y=true_travel_times_A2_A1.detach().cpu().numpy(),
        ax=axs[1],
        cmap="Blues",
    )
    sns.kdeplot(
        x=posterior_samples["travel_time_A1_A2"].detach().cpu().numpy(),
        y=posterior_samples["travel_time_A2_A1"].detach().cpu().numpy(),
        ax=axs[1],
        cmap="Reds",
    )
    axs[1].set_xlabel("Travel time A1 -> A2")
    axs[1].set_ylabel("Travel time A2 -> A1")
    axs[1].set_xlim(0, 5)
    axs[1].set_ylim(0, 5)
    handles = [
        mpatches.Patch(facecolor=plt.cm.Reds(100), label="SVI Estimated Posterior"),
        mpatches.Patch(facecolor=plt.cm.Blues(100), label="Ground Truth"),
    ]
    plt.legend(handles=handles)

    plt.show()


if __name__ == "__main__":
    main()
