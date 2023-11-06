import unittest

import pandas as pd

from bayes_air.schedule import parse_flight, parse_schedule
from bayes_air.types import Airport, Flight


class TestFlightParsing(unittest.TestCase):
    def test_parse_flight(self):
        schedule_row = (
            "F123",
            "ABC",
            "XYZ",
            0.0,
            1.0,
            0.5,
            1.5,
        )

        expected_flight = Flight(
            flight_number="F123",
            origin="ABC",
            destination="XYZ",
            scheduled_departure_time=0.0,
            scheduled_arrival_time=1.0,
            actual_departure_time=0.5,
            actual_arrival_time=1.5,
        )

        result_flight = parse_flight(schedule_row)
        self.assertEqual(result_flight, expected_flight)

    def test_parse_schedule(self):
        schedule_data = {
            "flight_number": ["F123", "F456"],
            "origin_airport": ["ABC", "XYZ"],
            "destination_airport": ["XYZ", "ABC"],
            "scheduled_departure_time": [0.0, 0.0],
            "scheduled_arrival_time": [1.0, 1.0],
            "actual_departure_time": [0.5, 0.5],
            "actual_arrival_time": [1.5, 1.5],
        }

        schedule_df = pd.DataFrame(schedule_data)
        expected_flights = [
            Flight(
                flight_number="F123",
                origin="ABC",
                destination="XYZ",
                scheduled_departure_time=0.0,
                scheduled_arrival_time=1.0,
                actual_departure_time=0.5,
                actual_arrival_time=1.5,
            ),
            Flight(
                flight_number="F456",
                origin="XYZ",
                destination="ABC",
                scheduled_departure_time=0.0,
                scheduled_arrival_time=1.0,
                actual_departure_time=0.5,
                actual_arrival_time=1.5,
            ),
        ]
        expected_airports = [Airport("ABC"), Airport("XYZ")]

        result_flights, result_airports = parse_schedule(schedule_df)
        self.assertEqual(result_flights, expected_flights)
        self.assertEqual(result_airports, expected_airports)


if __name__ == "__main__":
    unittest.main()
