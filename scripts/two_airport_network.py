"""Run the simulation for a simple two-airport network."""
import pandas as pd
import pyro
import torch

from bayes_air.model import air_traffic_network_model
from bayes_air.network import NetworkState
from bayes_air.schedule import parse_schedule


def main():
    pyro.enable_validation(True)
    pyro.set_rng_seed(1)

    # Define a simple schedule that sends the aircraft from airport 0 to airport 1 and back
    # Time is defined in hours from the start of the simulation
    schedule = pd.DataFrame(
        {
            "flight_number": ["F1", "F1"],
            "origin_airport": ["A1", "A2"],
            "destination_airport": ["A2", "A1"],
            "scheduled_departure_time": [torch.tensor(0.0), torch.tensor(1.0)],
            "scheduled_arrival_time": [torch.tensor(1.0), torch.tensor(3.0)],
            "actual_departure_time": [None, None],
            "actual_arrival_time": [None, None],
        }
    )

    flights, airports = parse_schedule(schedule)
    airports[0].available_aircraft.append(torch.tensor(0.0))
    state = NetworkState(
        airports={airport.code: airport for airport in airports},
        pending_flights=flights,
    )

    model_graph = pyro.render_model(
        air_traffic_network_model,
        model_args=(state, 5.0, 0.1),
        render_params=True,
        render_distributions=True,
    )
    model_graph.render(view=True)
    


if __name__ == "__main__":
    main()
