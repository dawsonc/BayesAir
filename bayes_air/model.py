"""Define a probabilistic model for an air traffic network."""
from copy import deepcopy

import pyro
import pyro.distributions as dist
import torch

from bayes_air.network import NetworkState
from bayes_air.types import QueueEntry

FAR_FUTURE_TIME = 30.0


def air_traffic_network_model(
    states: list[NetworkState],
    delta_t: float = 0.1,
    max_t: float = FAR_FUTURE_TIME,
    device=None,
    include_cancellations: bool = False,
):
    """
    Simulate the behavior of an air traffic network.

    Args:
        states: the starting states of the simulation (will run an independent
            simulation from each start state). All states must include the same
            airports.
        delta_t: the time resolution of the simulation, in hours
        max_t: the maximum time to simulate, in hours
        device: the device to run the simulation on
        include_cancellations: whether to include the possibility of flight
            cancellations (if False, crew/aircraft reserves will not be modeled)
    """
    if device is None:
        device = torch.device("cpu")

    # Copy state to avoid modifying it
    states = deepcopy(states)

    # Define system-level parameters
    runway_use_time_std_dev = pyro.param(
        "runway_use_time_std_dev",
        torch.tensor(0.1),  # used to be 0.025
        constraint=dist.constraints.positive,
    )
    travel_time_variation = pyro.param(
        "travel_time_variation",
        torch.tensor(0.1),  # used to be 0.05
        constraint=dist.constraints.positive,
    )
    turnaround_time_variation = pyro.param(
        "turnaround_time_variation",
        torch.tensor(0.1),  # used to be 0.05
        constraint=dist.constraints.positive,
    )

    # Sample latent variables for airports.
    airport_codes = states[0].airports.keys()
    airport_turnaround_times = {
        code: pyro.sample(f"{code}_mean_turnaround_time", dist.Gamma(1.0, 2.0))
        for code in airport_codes
    }
    airport_service_times = {
        code: pyro.sample(f"{code}_mean_service_time", dist.Gamma(1.5, 10.0))
        for code in airport_codes
    }
    travel_times = {
        (origin, destination): pyro.sample(
            f"travel_time_{origin}_{destination}", dist.Gamma(4.0, 1.25)
        )
        for origin in airport_codes
        for destination in airport_codes
        if origin != destination
    }

    if include_cancellations:
        airport_initial_available_aircraft = {
            code: torch.exp(
                pyro.sample(
                    f"{code}_log_initial_available_aircraft", dist.Normal(0.0, 1.0)
                )
            )
            for code in airport_codes
        }
    else:
        # To ignore cancellations, just provide practcially infinite reserves
        airport_initial_available_aircraft = {
            code: torch.tensor(1000.0, device=device) for code in airport_codes
        }

    # Simulate for each state
    output_states = []
    # for day_ind in pyro.plate("days", len(states)):
    for day_ind in pyro.markov(range(len(states)), history=1):
        state = states[day_ind]
        var_prefix = f"day{day_ind}_"

        # print(f"============= Starting day {day_ind} =============")
        # print(f"# pending flights: {len(state.pending_flights)}")
        # print(f"Initial aircraft: {airport_initial_available_aircraft}")
        # print(f"# in-transit flights: {len(state.in_transit_flights)}")
        # print(f"# completed flights: {len(state.completed_flights)}")
        # print("Travel times:")
        # print(travel_times)

        # Assign the latent variables to the airports
        for airport in state.airports.values():
            airport.mean_service_time = airport_service_times[airport.code]
            airport.runway_use_time_std_dev = runway_use_time_std_dev
            airport.mean_turnaround_time = airport_turnaround_times[airport.code]
            airport.turnaround_time_std_dev = (
                turnaround_time_variation * airport.mean_turnaround_time
            )

            # Initialize the available aircraft list
            airport.num_available_aircraft = airport_initial_available_aircraft[
                airport.code
            ]
            i = 0
            while i < airport.num_available_aircraft:
                airport.available_aircraft.append(torch.tensor(0.0))
                i += 1

        # Simulate the movement of aircraft within the system for a fixed period of time
        t = 0.0
        while not state.complete:
            # Update the current time
            t += delta_t

            # All parked aircraft that are ready to turnaround get serviced
            for airport in state.airports.values():
                airport.update_available_aircraft(t)

            # If the maximum time has elapsed, add lots of reserve aircraft at each
            # airport. This is artificial and only done to ensure that the simulation
            # terminates.
            if t >= max_t:
                # print(f"TIME'S UP! Adding reserve aircraft at time {t}")
                for airport in state.airports.values():
                    airport.num_available_aircraft = airport.num_available_aircraft + 1
                    airport.available_aircraft.append(torch.tensor(t))

            # All flights that are able to depart get moved to the runway queue at their
            # origin airport
            ready_to_depart_flights, ready_times = state.pop_ready_to_depart_flights(
                t, var_prefix
            )
            for flight, ready_time in zip(ready_to_depart_flights, ready_times):
                queue_entry = QueueEntry(flight=flight, queue_start_time=ready_time)
                state.airports[flight.origin].runway_queue.append(queue_entry)

            # All flights that are using the runway get serviced
            for airport in state.airports.values():
                departed_flights, landing_flights = airport.update_runway_queue(
                    t, var_prefix
                )

                # Departing flights get added to the in-transit list, while landed flights
                # get added to the completed list
                state.add_in_transit_flights(
                    departed_flights, travel_times, travel_time_variation, var_prefix
                )
                state.add_completed_flights(landing_flights)

            # All flights that are in transit get moved to the runway queue at their
            # destination airport, if enough time has elapsed
            state.update_in_transit_flights(t)

        # print(f"---------- Completing day {day_ind} ----------")
        # print(f"# pending flights: {len(state.pending_flights)}")
        # print(f"# in-transit flights: {len(state.in_transit_flights)}")
        # print(f"# completed flights: {len(state.completed_flights)}")

        # Once we're done, return the state (this will include the actual arrival/departure
        # times for each aircraft)
        output_states.append(state)

    return output_states
