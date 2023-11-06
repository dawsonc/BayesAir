"""Define the network model"""
from dataclasses import dataclass, field
from typing import Union

import pyro
import pyro.distributions as dist
import torch

from bayes_air.types import QueueEntry
from bayes_air.types import Airport, AirportCode, Flight, Time


@dataclass
class NetworkState:
    """The state of the network at a given time.

    Attributes:
        airports: A dictionary mapping airport codes to Airport objects.
        pending_flights: A list of flights that have not yet departed, sorted by
            scheduled departure time.
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

    def pop_ready_to_depart_flights(self, time: Time) -> list[Flight]:
        """Pop all flights from the pending flights list that are able to depart.

        Modifies the pending_flights list.

        Args:
            time: The time at which to check for flights ready to depart.

        Returns:
            A list of flights that are ready to depart.
            A list of times at which those flights became ready
        """
        # The pending flights list is already sorted by scheduled departure time, so we
        # can just iterate through it, removing flights that are ready to depart
        ready_to_depart_flights = []
        ready_times = []
        new_pending_flights = []
        for flight in self.pending_flights:
            # A flight is ready to depart if its scheduled departure time is less
            # than the current time AND its source airport has an available aircraft
            if (
                flight.scheduled_departure_time <= time
                and self.airports[flight.origin].num_available_aircraft > 0
            ):
                # Remove the aircraft from the airport
                turnaround_time = self.airports[flight.origin].available_aircraft.pop(0)

                # Add the flight to the list of flights ready to depart
                ready_to_depart_flights.append(flight)

                # Get the time at which this flight was ready to depart (the later
                # of the scheduled departure time and the turnaround ready time)
                ready_times.append(
                    torch.maximum(turnaround_time, flight.scheduled_departure_time)
                )
            else:
                # Add the flight to the new pending flights list
                new_pending_flights.append(flight)

        # Update the pending flights list
        self.pending_flights = new_pending_flights

        return ready_to_depart_flights, ready_times

    def add_in_transit_flights(
        self,
        departing_flights: list[Flight],
        travel_times: dict[tuple[AirportCode, AirportCode], Time],
        travel_time_variation: Union[torch.tensor, float],
    ) -> None:
        """Add a list of flights to the in-transit flights list.

        Args:
            departing_flights: The list of flights to add.
            travel_times: A dictionary mapping origin-destination pairs to nominal
                travel times.
            travel_time_variation: The fractional variation in travel time.
        """
        # For each departing flight, sample a travel time and then add it to the
        # in-transit flights list
        for flight in departing_flights:
            # Sample a travel time
            nominal_travel_time = travel_times[flight.origin, flight.destination]
            var_name = str(flight) + "_travel_time"
            travel_time = pyro.sample(
                var_name,
                dist.Normal(
                    nominal_travel_time, nominal_travel_time * travel_time_variation
                ),
            )

            # Add the flight to the in-transit flights list
            self.in_transit_flights.append((flight, flight.actual_departure_time + travel_time))

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
            else:
                new_in_transit_flights.append((flight, arrival_time))

        # Update the in-transit flights list
        self.in_transit_flights = new_in_transit_flights
