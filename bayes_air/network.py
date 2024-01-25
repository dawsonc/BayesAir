"""Define the network model"""
from dataclasses import dataclass, field
from typing import Union

import pyro
import pyro.distributions as dist
import torch

from bayes_air.types import Airport, AirportCode, Flight, QueueEntry, Time


@dataclass
class NetworkState:
    """The state of the network at a given time.

    Attributes:
        airports: A dictionary mapping airport codes to Airport objects.
        pending_flights: A list of flights that have not yet departed, sorted by
            scheduled departure time. Will be sorted after initialization.
        in_transit_flights: A list of flights that are currently in the air, along with
            the time at which they will arrive at their destination.
        completed_flights: A list of flights that have completed their journeys.
    """

    airports: dict[AirportCode, Airport]
    pending_flights: list[Flight]
    in_transit_flights: list[tuple[Flight, Time]] = field(default_factory=list)
    completed_flights: list[Flight] = field(default_factory=list)

    def __post_init__(self):
        # Sort pending flights by scheduled departure time
        self.pending_flights.sort(key=lambda flight: flight.scheduled_departure_time)

    @property
    def complete(self):
        """Return True if all flights have completed their journeys."""
        return (
            len(self.pending_flights) == 0
            and len(self.in_transit_flights) == 0
            and all([len(a.runway_queue) == 0 for a in self.airports.values()])
        )

    def pop_ready_to_depart_flights(
        self, time: Time, var_prefix: str = ""
    ) -> list[Flight]:
        """Pop all flights from the pending flights list that are able to depart.

        Modifies the pending_flights list.

        Args:
            time: The time at which to check for flights ready to depart.

        Returns:
            A list of flights that are ready to depart.
            A list of times at which those flights became ready
        """
        # Check the flights departing from each airport
        ready_to_depart_flights_by_airport = {}
        new_pending_flights = []
        for flight in self.pending_flights:
            if flight.scheduled_departure_time <= time:
                if flight.origin not in ready_to_depart_flights_by_airport:
                    ready_to_depart_flights_by_airport[flight.origin] = []

                # print(f"{flight} ready to depart")
                ready_to_depart_flights_by_airport[flight.origin].append(flight)
            else:
                new_pending_flights.append(flight)

        # For each airport, check if there are enough available aircraft to depart
        ready_to_depart_flights = []
        ready_times = []
        for airport_code, flights in ready_to_depart_flights_by_airport.items():
            airport = self.airports[airport_code]
            num_available_aircraft = airport.num_available_aircraft
            num_flights_to_depart = len(flights)

            # Cancel some number of flights if there are not enough available aircraft
            cancellation_probability = 1 - torch.maximum(
                torch.minimum(
                    torch.tensor(1.0, device=time.device),
                    num_available_aircraft / num_flights_to_depart,
                ),
                torch.tensor(0.0, device=time.device),
            )

            # print(
            #     f"{airport.code} has {num_available_aircraft} available aircraft; {num_flights_to_depart} flights ready to depart"
            # )
            # print(f"Cancel probability: {cancellation_probability}")

            for flight in flights:
                if flight.simulated_cancelled is None:
                    var_name = var_prefix + str(flight) + "_cancelled"
                    flight.simulated_cancelled = pyro.sample(
                        var_name,
                        dist.RelaxedBernoulliStraightThrough(
                            torch.tensor(0.5),  # temperature
                            probs=cancellation_probability,
                        ),
                        obs=flight.actually_cancelled,
                    )

                if flight.simulated_cancelled == 0:
                    if airport.num_available_aircraft <= 0:
                        # print(f"{flight} delayed")
                        new_pending_flights.append(flight)
                    else:
                        # print(f"{flight} departs")
                        ready_to_depart_flights.append(flight)
                        airport.num_available_aircraft = (
                            airport.num_available_aircraft - 1
                        )
                        aircraft_turnaround_t = airport.available_aircraft.pop(0)
                        ready_times.append(
                            torch.maximum(
                                aircraft_turnaround_t, flight.scheduled_departure_time
                            )
                        )
                else:
                    # print(f"{flight} cancelled")
                    self.completed_flights.append(flight)

        # Update the pending flights list
        self.pending_flights = new_pending_flights

        return ready_to_depart_flights, ready_times

    def add_in_transit_flights(
        self,
        departing_flights: list[Flight],
        travel_times: dict[tuple[AirportCode, AirportCode], Time],
        travel_time_variation: Union[torch.tensor, float],
        var_prefix: str = "",
    ) -> None:
        """Add a list of flights to the in-transit flights list.

        Args:
            departing_flights: The list of flights to add.
            travel_times: A dictionary mapping origin-destination pairs to nominal
                travel times.
            travel_time_variation: The fractional variation in travel time.
            var_prefix: prefix for sampled variable names.
        """
        # For each departing flight, sample a travel time and then add it to the
        # in-transit flights list
        for flight in departing_flights:
            # Sample a travel time
            nominal_travel_time = travel_times[flight.origin, flight.destination]
            var_name = var_prefix + str(flight) + "_travel_time"
            travel_time = pyro.sample(
                var_name,
                dist.Normal(
                    nominal_travel_time, nominal_travel_time * travel_time_variation
                ),
            )

            # Add the flight to the in-transit flights list
            self.in_transit_flights.append(
                (flight, flight.simulated_departure_time + travel_time)
            )

    def add_completed_flights(self, landing_flights: list[Flight]) -> None:
        """Add a list of flights to the completed flights list.

        Args:
            landing_flights: The list of flights to add.
        """
        self.completed_flights.extend(landing_flights)

    def update_in_transit_flights(self, time: Time) -> None:
        """Update all in-transit flights that are ready to arrive.

        Args:
            time: The current time.
        """
        new_in_transit_flights = []

        for flight, arrival_time in self.in_transit_flights:
            # If the flight is ready to arrive, add it to the runway queue
            if arrival_time <= time:
                # Add the completed flights to the arrival queue
                queue_entry = QueueEntry(flight=flight, queue_start_time=arrival_time)
                self.airports[flight.destination].runway_queue.append(queue_entry)
                # print(f"\t{flight} landing at {arrival_time}")
            else:
                new_in_transit_flights.append((flight, arrival_time))
                # print(f"\t{flight} still in transit; will land {arrival_time}")

        # Update the in-transit flights list
        self.in_transit_flights = new_in_transit_flights
