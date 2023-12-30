"""Define types for flights in the network."""
from dataclasses import dataclass
from typing import Optional

import torch

from bayes_air.types.util import AirportCode, Time


@dataclass
class Flight:
    """A flight between two airports.

    Attributes:
        flight_number: The flight number.
        origin: The origin airport code.
        destination: The destination airport code.
        scheduled_departure_time: The scheduled departure time.
        scheduled_arrival_time: The scheduled arrival time.
        actually_cancelled: Whether the flight was cancelled in reality (0 or 1).
        simulated_cancelled: Whether the flight was cancelled in the simulation (0 or 1).
        simulated_departure_time: The simulated departure time.
        simulated_arrival_time: The simulated arrival time.
        actual_departure_time: The actual departure time.
        actual_arrival_time: The actual arrival time.
        wheels_on_time: The time the wheels touched the ground.
        wheels_off_time: The time the wheels left the ground.
    """

    flight_number: str
    origin: AirportCode
    destination: AirportCode
    scheduled_departure_time: Time
    scheduled_arrival_time: Time
    actually_cancelled: torch.tensor
    simulated_cancelled: Optional[torch.tensor] = None
    simulated_departure_time: Optional[Time] = None
    simulated_arrival_time: Optional[Time] = None
    actual_departure_time: Optional[Time] = None
    actual_arrival_time: Optional[Time] = None
    wheels_on_time: Optional[Time] = None
    wheels_off_time: Optional[Time] = None

    def __str__(self) -> str:
        return f"{self.flight_number}_{self.origin}_{self.destination}"
