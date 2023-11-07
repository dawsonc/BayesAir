"""Define types for flights in the network."""
from dataclasses import dataclass
from typing import Optional

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
        actual_departure_time: The actual departure time.
        actual_arrival_time: The actual arrival time.
    """

    flight_number: str
    origin: AirportCode
    destination: AirportCode
    scheduled_departure_time: Time
    scheduled_arrival_time: Time
    actual_departure_time: Optional[Time] = None
    actual_arrival_time: Optional[Time] = None

    def __str__(self) -> str:
        return f"{self.flight_number}_{self.origin}_{self.destination}"
