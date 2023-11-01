"""Define a probabilistic model for an air traffic network."""
import copy

import pyro
import pyro.distributions as dist
import torch

from bayes_air.state import State


def air_traffic_network_model(state: State, T: float = 24.0, delta_t: float = 0.1):
    """
    Simulate the behavior of one aircraft flying between two airports.

    Data should be sorted in order of increasing scheduled departure time.

    Args:
        state: the starting state of the simulation
        T: the duration of the simulation, in hours
        delta_t: the time resolution of the simulation, in hours
    """
    # Define parameters used by the simulation
    runway_use_time_standard_deviation = 1.0 / 60  # 1 minute
    travel_time_fractional_variation = 0.05  # 5% variation in travel time
    turnaround_time_fractional_variation = 0.05  # 5% variation in turnaround time

    # Sample latent variables for airports
    airport_codes = state.airports.keys()
    airport_turnaround_times = {
        code: pyro.sample(f"{code}_mean_turnaround_time", dist.Normal(0.75, 0.25))
        for code in airport_codes
    }
    airport_service_times = {
        code: pyro.sample(f"{code}_mean_service_time", dist.Normal(9.0 / 60, 3.0 / 60))
        for code in airport_codes
    }
    travel_times = {
        (origin, destination): pyro.sample(
            f"travel_time_{origin}_{destination}", dist.Normal(1.0, 0.25)
        )
        for origin in airport_codes
        for destination in airport_codes
        if origin != destination
    }
    crew_availability = {
        code: pyro.sample(f"{code}_initial_crew_availability", dist.Normal(20, 5))
        for code in airport_codes
    }

    # Make a copy of the state so we don't change the input state
    state = copy.deepcopy(state)

    # Simulate the movement of aircraft and people within the system for a fixed period of time
    t = 0.0

    #Initializing all available crew by sampling the latent crew parameter
    for code in airport_codes:
        airport.number_of_available_crew = crew_availability[code]

    for _ in pyro.markov(range(int(T // delta_t))):
        # Update the current time
        t += delta_t

        # Parked aircraft that are ready to depart move to the runway service queue.
        # An aircraft is ready to depart if the current time is greater than its
        # earliest possible departure time and if there is sufficient crew.
        for airport in state.airports.values():
            new_parked_aircraft = []
            for aircraft, earliest_departure_time in airport.parked_aircraft:
                #Checking if there is sufficient crew
                if t >= earliest_departure_time and airport.number_of_available_crew >= 3:
                    # logging.debug(f"(t={t:.2f}) {aircraft} is ready to depart from {airport.code}; moving to queue")
                    # TODO can queue items be a dataclass?
                    airport.runway_queue.append(
                        [
                            aircraft,
                            True,  # departing
                            earliest_departure_time,  # queue start time
                            0.0,  # no wait time yet
                            None,  # no service time yet
                        ]
                    )
                    #Removing the number of available crew from the integer value
                    airport.number_of_available_crew -= 3
                # TODO is there a way to not add it to the list?
                else:
                    new_parked_aircraft.append((aircraft, earliest_departure_time))

            airport.parked_aircraft = new_parked_aircraft

        # In-flight aircraft that are ready to arrive move to the service queue.
        # An aircraft is ready to arrive if the current time is greater than its
        # earliest possible arrival time.
        new_in_flight_aircraft = []
        for aircraft, earliest_arrival_time in state.in_flight_aircraft:
            if t >= earliest_arrival_time:
                # logging.debug(f"(t={t:.2f}) {aircraft} is ready to arrive at {aircraft.current_leg.destination}; moving to queue")
                destination_airport = aircraft.current_leg.destination
                state.airports[destination_airport].runway_queue.append(
                    [
                        aircraft,
                        False,  # arriving
                        earliest_arrival_time,  # queue start time
                        0.0,  # no wait time yet
                        None,  # no service time yet
                    ]
                )
            else:
                new_in_flight_aircraft.append((aircraft, earliest_arrival_time))

        state.in_flight_aircraft = new_in_flight_aircraft

        # Aircraft in the runway queue that are ready to be serviced are serviced.
        for airport in state.airports.values():
            # Process the aircraft at the start of the queue as long as it's not waiting
            # past the current time.

            # TODO this is a lot of nested loops/conditionals. Consider factoring into
            # smaller functions.
            while airport.runway_queue and (
                airport.runway_queue[0][4] is None or t >= airport.runway_queue[0][4]
            ):
                # If the aircraft at the front of the queue does not have a service time
                # yet, sample one from the airport's service time distribution. If we have
                # observed the actual arrival/departure time, use that instead.
                if airport.runway_queue[0][4] is None:
                    (
                        aircraft,
                        departing,
                        queue_start_time,
                        total_wait_time,
                        _,
                    ) = airport.runway_queue[0]

                    var_name = str(aircraft)
                    if departing:
                        var_name += "_departure_service_time"
                    else:
                        var_name += "_arrival_service_time"

                    service_time = pyro.sample(
                        var_name,
                        dist.Exponential(
                            torch.tensor(1.0 / airport_service_times[airport.code])
                        ),
                    )
                    # TODO fix this tensor thing.
                    print(
                        f"Sampling service time {service_time} with rate {1.0 / airport_service_times[airport.code]}"
                    )
                    airport.runway_queue[0] = [
                        aircraft,
                        departing,
                        queue_start_time,
                        total_wait_time,
                        t + service_time,
                    ]

                    # Add the service time to the wait time of every aircraft in the queue
                    for i in range(len(airport.runway_queue)):
                        airport.runway_queue[i][3] += service_time

                    # logging.debug(f"(t={t:.2f}) {aircraft} is at the front of the queue at {airport.code}; assigning service time {t + service_time:.2f}")

                # If the aircraft at the front of the queue has been serviced, it takes off
                # or lands.
                if t >= airport.runway_queue[0][4]:
                    (
                        aircraft,
                        departing,
                        queue_start_time,
                        total_wait_time,
                        service_time,
                    ) = airport.runway_queue.pop(0)

                    # logging.debug(f"(t={t:.2f}) {aircraft} is ready to be served at {airport.code}")

                    var_name = str(aircraft)
                    if departing:
                        # The aircraft takes off.

                        # Record its actual departure time. If we have observed the actual
                        # departure time, use that instead to condition the model.
                        aircraft.current_leg.actual_departure_time = pyro.sample(
                            var_name + "_actual_departure_time",
                            dist.Normal(
                                queue_start_time + total_wait_time,
                                runway_use_time_standard_deviation,
                            ),
                            obs=aircraft.current_leg.actual_departure_time,
                        )

                        # Move the aircraft to the in-flight list, sampling a travel time
                        # from the travel time distribution.
                        nominal_travel_time = travel_times[
                            aircraft.current_leg.origin,
                            aircraft.current_leg.destination,
                        ]
                        travel_time = pyro.sample(
                            var_name + "_travel_time",
                            dist.Normal(
                                nominal_travel_time,
                                nominal_travel_time * travel_time_fractional_variation,
                            ),
                        )
                        state.in_flight_aircraft.append(
                            (
                                aircraft,
                                aircraft.current_leg.actual_departure_time
                                + travel_time,
                            )
                        )

                        # logging.debug(f"\t{aircraft} assigned travel time {travel_time:.2f}; will arrive at {aircraft.current_leg.destination} at {t + travel_time:.2f}")
                    else:
                        # The aircraft lands.

                        # Record its actual arrival time. If we have observed the actual
                        # arrival time, use that instead to condition the model.
                        aircraft.current_leg.actual_arrival_time = pyro.sample(
                            var_name + "_actual_arrival_time",
                            dist.Normal(
                                queue_start_time + total_wait_time,
                                runway_use_time_standard_deviation,
                            ),
                            obs=aircraft.current_leg.actual_arrival_time,
                        )

                        # Sample a random turnaround time from the distribution for the
                        # destination airport.
                        destination = aircraft.current_leg.destination
                        nominal_turnaround_time = airport_turnaround_times[destination]
                        turnaround_time = pyro.sample(
                            var_name + "_turnaround_time",
                            dist.Normal(
                                nominal_turnaround_time,
                                nominal_turnaround_time
                                * turnaround_time_fractional_variation,
                            ),
                        )
                        # The earliest the aircraft can depart is the current time plus the
                        # turnaround time or its next scheduled departure time, whichever is
                        # later.
                        next_scheduled_departure_time = (
                            aircraft.next_leg.scheduled_departure_time
                            if aircraft.next_leg is not None
                            else float("inf")
                        )
                        earliest_departure_time = max(
                            aircraft.current_leg.actual_arrival_time + turnaround_time,
                            next_scheduled_departure_time,
                        )
                        state.airports[destination].parked_aircraft.append(
                            (aircraft, earliest_departure_time)
                        )

                        # logging.debug(f"\t{aircraft} assigned turnaround {turnaround_time:.2f}; will depart at {earliest_departure_time:.2f}")

                        # Mark the aircraft as being on the next leg of its itinerary
                        aircraft.advance_itinerary()

                        #Add crew repository back when it has been serviced from the airport queue after it lands.
                        airport.number_of_available_crew += 3

    # Once we're done, return the state (this will include the actual arrival/departure
    # times for each aircraft)
    return state
