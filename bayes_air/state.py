"""Define types and methods for working with the state of a simulation"""
from dataclasses import dataclass

import pandas as pd

from bayes_air.aircraft import Aircraft
from bayes_air.airport import Airport
from bayes_air.schedule import parse_itinerary_item
from bayes_air.util import AirportCode, Time


@dataclass
class State:
    in_flight_aircraft: list[tuple[Aircraft, Time]]  # aircraft, earliest arrival time
    airports: dict[AirportCode, Airport]


def parse_schedule(schedule_df: pd.DataFrame) -> State:
    """Parse a pandas dataframe for a schedule into a state object.

    Args:
        schedule_df: A pandas dataframe with the following columns:
            tail_number: The tail number of the aircraft
            origin_airport: The airport code of the origin airport
            destination_airport: The airport code of the destination airport
            scheduled_departure_time: The scheduled departure time
            scheduled_arrival_time: The scheduled arrival time
            actual_departure_time: The actual departure time
            actual_arrival_time: The actual arrival time
    """
    # Create an airport object for each airport in the schedule
    airport_codes = pd.concat(
        [schedule_df["origin_airport"], schedule_df["destination_airport"]]
    ).unique()
    airports = {
        airport_code: Airport(code=airport_code, runway_queue=[], parked_aircraft=[], number_of_available_crew = 0)
        for airport_code in airport_codes
    }

    # Construct objects for each aircraft, extracting their itineraries, and parking
    # them in their origin airports (assuming no turnaround time for the first flight of
    # the day)
    for tail_number in schedule_df["tail_number"].unique():
        # Get the itinerary for this aircraft
        itinerary_df = schedule_df[
            schedule_df["tail_number"] == tail_number
        ].sort_values(by="scheduled_departure_time")
        itinerary_items = [
            parse_itinerary_item(row)
            for row in zip(
                itinerary_df["origin_airport"],
                itinerary_df["destination_airport"],
                itinerary_df["scheduled_departure_time"],
                itinerary_df["scheduled_arrival_time"],
                itinerary_df["actual_departure_time"],
                itinerary_df["actual_arrival_time"],
            )
        ]

        # Create the aircraft object and park it at the first airport
        aircraft = Aircraft(tail_number=tail_number, itinerary=itinerary_items)
        origin_airport = airports[itinerary_items[0].origin]
        origin_airport.parked_aircraft.append(
            (aircraft, itinerary_items[0].scheduled_departure_time)
        )

    # No aircraft start in flight
    in_flight_aircraft = []

    return State(in_flight_aircraft=in_flight_aircraft, airports=airports)
