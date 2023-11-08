"""Test the bayes_air.types module."""
import unittest

from bayes_air.types import Airport, Flight, Time
from bayes_air.types.airport import QueueEntry


class TestAirport(unittest.TestCase):
    def setUp(self):
        # Create an instance of the Airport class for testing
        self.airport = Airport(
            code="ABC",
            available_aircraft=[Time(0.0), Time(0.0), Time(0.0)],
            mean_service_time=Time(10.0),
            runway_use_time_std_dev=Time(2.0),
            mean_turnaround_time=Time(30.0),
            turnaround_time_std_dev=Time(5.0),
        )

    def test_update_available_aircraft(self):
        # If there are no aircraft in the turnaround queue, the number of available
        # aircraft should not change when we update
        self.airport.update_available_aircraft(0.0)
        self.assertEqual(self.airport.num_available_aircraft, 3)

        # If there is an aircraft in the turnaround queue that is ready to turnaround,
        # the number of available aircraft should increase by 1 when we update
        self.airport.turnaround_queue.append(10.0)
        self.airport.update_available_aircraft(15.0)
        self.assertEqual(self.airport.num_available_aircraft, 4)

        # If there is an aircraft in the turnaround queue that is not ready to
        # turnaround, the number of available aircraft should not change when we update
        self.airport.turnaround_queue.append(20.0)
        self.airport.update_available_aircraft(15.0)
        self.assertEqual(self.airport.num_available_aircraft, 4)

    def test_update_runway_queue_takeoff(self):
        # Assuming a flight is in the runway queue and ready for takeoff
        flight = Flight(
            flight_number="F123",
            origin="ABC",
            destination="XYZ",
            scheduled_departure_time=10.0,
            scheduled_arrival_time=20.0,
        )
        queue_entry = QueueEntry(flight, 0.0, assigned_service_time=Time(10.0))
        self.airport.runway_queue.append(queue_entry)

        # Update runway queue and check the departed flights
        departed_flights, landed_flights = self.airport.update_runway_queue(10.0)
        self.assertEqual(len(departed_flights), 1)
        self.assertEqual(departed_flights[0], flight)
        self.assertEqual(len(landed_flights), 0)

    def test_update_runway_queue_landing(self):
        # Assuming a flight is in the runway queue and ready for landing
        flight = Flight(
            flight_number="F123",
            origin="XYZ",
            destination="ABC",
            scheduled_departure_time=10.0,
            scheduled_arrival_time=20.0,
        )
        queue_entry = QueueEntry(flight, 0.0, assigned_service_time=Time(20.0))
        self.airport.runway_queue.append(queue_entry)

        # Update runway queue and check the departed flights
        departed_flights, landed_flights = self.airport.update_runway_queue(20.0)
        self.assertEqual(len(departed_flights), 0)
        self.assertEqual(len(landed_flights), 1)
        self.assertEqual(landed_flights[0], flight)


if __name__ == "__main__":
    unittest.main()
