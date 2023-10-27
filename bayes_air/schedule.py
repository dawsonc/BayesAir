"""Define types and methods for working with schedules."""
from dataclasses import dataclass
from typing import Optional

from bayes_air.components.types import AirportCode, Time


@dataclass
class ItineraryItem:
    origin: AirportCode
    destination: AirportCode
    scheduled_departure_time: Time
    scheduled_arrival_time: Time
    actual_departure_time: Optional[Time] = None
    actual_arrival_time: Optional[Time] = None


# Parse the provided data into our custom data structures
def parse_itinerary_item(schedule_row: tuple) -> ItineraryItem:
    """
    Parse a row of the schedule into an ItineraryItem object.

    Args:
        schedule_row: a tuple of the following items
            - origin_airport
            - destination_airport
            - scheduled_departure_time
            - scheduled_arrival_time
            - actual_departure_time
            - actual_arrival_time
    """
    (
        origin_airport,
        destination_airport,
        scheduled_departure_time,
        scheduled_arrival_time,
        actual_departure_time,
        actual_arrival_time,
    ) = schedule_row

    return ItineraryItem(
        origin=origin_airport,
        destination=destination_airport,
        scheduled_departure_time=scheduled_departure_time,
        scheduled_arrival_time=scheduled_arrival_time,
        actual_departure_time=actual_departure_time,
        actual_arrival_time=actual_arrival_time,
    )
