"""Define methods for working with schedules."""
import pandas as pd
import torch

from bayes_air.types import Airport, Flight, Time


# Parse the provided data into our custom data structures
def parse_flight(schedule_row: tuple) -> Flight:
    """
    Parse a row of the schedule into an Flight object.

    Args:
        schedule_row: a tuple of the following items
            - flight_number
            - origin_airport
            - destination_airport
            - cancelled
            - scheduled_departure_time
            - scheduled_arrival_time
            - actual_departure_time
            - actual_arrival_time
            - wheels_off_time
            - wheels_on_time
    """
    flight_number = schedule_row["flight_number"]
    origin_airport = schedule_row["origin_airport"]
    destination_airport = schedule_row["destination_airport"]
    scheduled_departure_time = schedule_row["scheduled_departure_time"]
    scheduled_arrival_time = schedule_row["scheduled_arrival_time"]
    actual_departure_time = schedule_row["actual_departure_time"]
    actual_arrival_time = schedule_row["actual_arrival_time"]
    wheels_off_time = schedule_row["wheels_off_time"]
    wheels_on_time = schedule_row["wheels_on_time"]
    cancelled = torch.tensor(1) if schedule_row["cancelled"] else torch.tensor(0)

    # If the flight was cancelled, set the measured times to None
    if schedule_row["cancelled"]:
        actual_departure_time = None
        actual_arrival_time = None
        wheels_off_time = None
        wheels_on_time = None

    return Flight(
        flight_number=flight_number,
        origin=origin_airport,
        destination=destination_airport,
        scheduled_departure_time=Time(scheduled_departure_time),
        scheduled_arrival_time=Time(scheduled_arrival_time),
        actually_cancelled=cancelled,
        actual_departure_time=Time(actual_departure_time)
        if actual_departure_time is not None
        else None,
        actual_arrival_time=Time(actual_arrival_time)
        if actual_arrival_time is not None
        else None,
        wheels_off_time=Time(wheels_off_time) if wheels_off_time is not None else None,
        wheels_on_time=Time(wheels_on_time) if wheels_on_time is not None else None,
    )


def parse_schedule(schedule_df: pd.DataFrame) -> tuple[list[Flight], list[Flight]]:
    """Parse a pandas dataframe for a schedule into a list of pending flights.

    Args:
        schedule_df: A pandas dataframe with the following columns:
            flight_number: The flight number
            origin_airport: The airport code of the origin airport
            destination_airport: The airport code of the destination airport
            scheduled_departure_time: The scheduled departure time
            scheduled_arrival_time: The scheduled arrival time
            actual_departure_time: The actual departure time
            actual_arrival_time: The actual arrival time
            wheels_off_time: The time the wheels left the ground
            wheels_on_time: The time the wheels touched the ground

    Returns:
        a list of flights, and
        a list of airports
    """
    # Get a list of flights
    flights = [parse_flight(row) for _, row in schedule_df.iterrows()]

    # Get a list of unique airport codes from the origin and destination columns
    airport_codes = pd.concat(
        [schedule_df["origin_airport"], schedule_df["destination_airport"]]
    ).unique()
    # Create an airport object for each airport code
    airports = [Airport(code) for code in airport_codes]

    return flights, airports
