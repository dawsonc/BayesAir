"""Define types for airports in the network."""
from dataclasses import dataclass, field
from typing import Optional

import pyro
import pyro.distributions as dist
import torch

from bayes_air.types.flight import Flight
from bayes_air.types.util import AirportCode, Time


@dataclass
class QueueEntry:
    """An entry in a runway queue.

    Attributes:
        flight: The flight associated with this queue entry.
        queue_start_time: The time at which the flight entered the queue.
        total_wait_time: The total duration the flight has spent in the queue.
        assigned_service_time: The time at which the flight will be serviced.
    """

    flight: Flight
    queue_start_time: Time
    total_wait_time: Time = field(default_factory=lambda: Time(0.0))
    assigned_service_time: Optional[Time] = None


@dataclass
class Airport:
    """Represents a single airport in the network.

    Attributes:
        code: The airport code.
        mean_service_time: The mean service time for departing and arriving aircraft
        runway_use_time_std_dev: The standard deviation of the runway use time for
            departing and arriving aircraft.
        mean_turnaround_time: The nominal turnaround time for aircraft landing at
            this airport.
        turnaround_time_std_dev: The standard deviation of the turnaround time for
            aircraft landing at this airport.
        runway_queue: The queue of aircraft waiting to take off or land.
        turnaround_queue: The queue of aircraft waiting to be cleaned/refueled/etc..
            Each entry in this queue represents a time at which the aircraft will be
            ready for its next departure.
        available_aircraft: a list of times at which aircraft became available (after
            turnaround).
        last_departure_time: The time at which the last aircraft departed.
    """

    code: AirportCode
    mean_service_time: Time = field(default_factory=lambda: Time(1.0))
    runway_use_time_std_dev: Time = field(default_factory=lambda: Time(1.0))
    mean_turnaround_time: Time = field(default_factory=lambda: Time(1.0))
    turnaround_time_std_dev: Time = field(default_factory=lambda: Time(1.0))
    num_available_aircraft: torch.tensor = field(
        default_factory=lambda: torch.tensor(0.0)
    )
    base_cancel_prob: torch.tensor = field(default_factory=lambda: torch.tensor(0.0))
    runway_queue: list[QueueEntry] = field(default_factory=list)
    turnaround_queue: list[Time] = field(default_factory=list)
    available_aircraft: list[Time] = field(default_factory=list)
    last_departure_time: Time = field(default_factory=lambda: Time(0.0))

    def update_available_aircraft(self, time: Time) -> None:
        """Update the number of available aircraft by checking the turnaround queue.

        Args:
            time: The current time.
        """
        new_turnaround_queue = []
        for turnaround_time in self.turnaround_queue:
            if turnaround_time <= time:
                # The aircraft is ready to depart, so add it to the available aircraft
                self.available_aircraft.append(turnaround_time)
                self.num_available_aircraft = self.num_available_aircraft + 1
            else:
                # The aircraft is not yet ready to depart, so add it to the new
                # turnaround queue
                new_turnaround_queue.append(turnaround_time)

        # Update the turnaround queue
        self.turnaround_queue = new_turnaround_queue

    def update_runway_queue(
        self, time: Time, var_prefix: str = ""
    ) -> tuple[list[Flight], list[Flight]]:
        """Update the runway queue by removing flights that have been serviced.

        Args:
            time: The current time.
            var_prefix: the prefix for sampled variable names

        Returns: a list of flights that have departed and a list of flights that have
            landed.
        """
        # While the flight at the front of the queue is ready to be serviced, service it
        departed_flights = []
        landed_flights = []
        while self.runway_queue and (
            self.runway_queue[0].assigned_service_time is None
            or self.runway_queue[0].assigned_service_time <= time
        ):
            # If no service time is assigned, assign one now by sampling from
            # the service time distribution
            if self.runway_queue[0].assigned_service_time is None:
                self._assign_service_time(self.runway_queue[0], time, var_prefix)

            # If the service time has elapsed, it takes off or lands
            if self.runway_queue[0].assigned_service_time <= time:
                queue_entry = self.runway_queue.pop(0)
                flight = queue_entry.flight

                departing = flight.origin == self.code
                if departing:
                    # Takeoff! Assign a departure time and add the flight to the
                    # list of departed flights
                    self._assign_departure_time(queue_entry, var_prefix)
                    departed_flights.append(flight)
                else:
                    # Landing! Assign an arrival time and add the aircraft to the
                    # turnaround queue
                    self._assign_arrival_time(queue_entry, var_prefix)
                    self._assign_turnaround_time(flight, var_prefix)
                    landed_flights.append(flight)

        return departed_flights, landed_flights

    def _assign_service_time(
        self, queue_entry: QueueEntry, time: Time, var_prefix: str = ""
    ) -> None:
        """Sample a random service time for an entry in the runway queue.

        Args:
            queue_entry: The queue entry to assign a service time to.
            time: The current time.
            var_prefix: prefix for sampled variable names.
        """
        departing = queue_entry.flight.origin == self.code
        var_name = var_prefix + str(queue_entry.flight)
        var_name += "_departure" if departing else "_arrival"
        var_name += "_service_time"
        service_time = pyro.sample(
            var_name,
            dist.Exponential(1.0 / self.mean_service_time.reshape(-1)),
        )

        # Update the waiting times for all aircraft
        for other_queue_entry in self.runway_queue:
            other_queue_entry.total_wait_time = (
                other_queue_entry.total_wait_time + service_time
            )

        # Update the time at which this aircraft leaves the queue
        queue_entry.assigned_service_time = (
            queue_entry.queue_start_time + queue_entry.total_wait_time
        )

        # print(
        #     f"\t{queue_entry.flight} assigned service time {queue_entry.assigned_service_time} (entered queue {queue_entry.queue_start_time} and sampled service time {service_time})"
        # )

    def _assign_departure_time(
        self,
        queue_entry: QueueEntry,
        var_prefix: str = "",
    ) -> None:
        """Sample a random departure time for a flight that is using the runway.

        Args:
            queue_entry: The queue entry for the flight to assign a departure time to.
            var_prefix: prefix for sampled variable names.
        """
        queue_entry.flight.simulated_departure_time = pyro.sample(
            var_prefix + str(queue_entry.flight) + "_simulated_departure_time",
            dist.Normal(
                queue_entry.queue_start_time + queue_entry.total_wait_time,
                self.runway_use_time_std_dev,
            ),
            obs=queue_entry.flight.actual_departure_time,
        )

        # print(
        #     f"\t{queue_entry.flight} departing at {queue_entry.flight.simulated_departure_time} (entered queue {queue_entry.queue_start_time} and waited {queue_entry.total_wait_time})"
        # )

    def _assign_arrival_time(
        self, queue_entry: QueueEntry, var_prefix: str = ""
    ) -> None:
        """Sample a random arrival time for a flight that is using the runway.

        Args:
            queue_entry: The queue entry for the flight to assign a arrival time to.
            var_prefix: prefix for sampled variable names.
        """
        queue_entry.flight.simulated_arrival_time = pyro.sample(
            var_prefix + str(queue_entry.flight) + "_simulated_arrival_time",
            dist.Normal(
                queue_entry.queue_start_time + queue_entry.total_wait_time,
                self.runway_use_time_std_dev,
            ),
            obs=queue_entry.flight.actual_arrival_time,
        )

        # print(
        #     f"\t{queue_entry.flight} arriving at {queue_entry.flight.simulated_arrival_time} (entered queue {queue_entry.queue_start_time} and waited {queue_entry.total_wait_time})"
        # )

    def _assign_turnaround_time(self, flight: Flight, var_prefix: str = "") -> None:
        """Sample a random turnaround time for an arrived aircraft.

        Args:
            flight: The flight to assign a turnaround time to.
            var_prefix: prefix for sampled variable names.
        """
        turnaround_time = pyro.sample(
            var_prefix + str(flight) + "_turnaround_time",
            dist.Normal(self.mean_turnaround_time, self.turnaround_time_std_dev),
        )
        self.turnaround_queue.append(flight.simulated_arrival_time + turnaround_time)
