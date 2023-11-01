"""Define a class representing an airport."""
from dataclasses import dataclass
from typing import Optional

from bayes_air.aircraft import Aircraft
# from bayes_air.crew import crew
from bayes_air.util import AirportCode, Time

@dataclass
class Airport:
    """
    Represents a single airport in the network.

    Attributes:
        code: The airport code.
        runway_queue: The queue of aircraft waiting to take off or land.
        parked_aircraft: The aircraft parked at the airport.
    """

    code: AirportCode
    runway_queue: list[
        list[Aircraft, bool, Time, Time, Optional[Time]]
    ]  # aircraft, departing/arriving, queue start time, total waiting time, assigned service time
    parked_aircraft: list[tuple[Aircraft, Time]]  # aircraft, earliest departure time
    number_of_available_crew: int #Integer value tracking number of available crew
    #crew: crew