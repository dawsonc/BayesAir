"""Define types used in the network simulation."""
from bayes_air.types.airport import Airport, QueueEntry
from bayes_air.types.flight import Flight
from bayes_air.types.util import AirportCode, Time

__all__ = [
    "AirportCode",
    "Time",
    "Flight",
    "Airport",
    "QueueEntry",
]
