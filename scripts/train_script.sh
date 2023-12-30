#!/bin/bash

# Train on nominal data
poetry run python scripts/wn_network.py --top-n 3 --days 2 --start-day 0 --svi-steps 2000 --plot-every 200 --svi-lr 0.01 &
poetry run python scripts/wn_network.py --top-n 3 --days 2 --start-day 2 --svi-steps 2000 --plot-every 200 --svi-lr 0.01 &
poetry run python scripts/wn_network.py --top-n 3 --days 2 --start-day 4 --svi-steps 2000 --plot-every 200 --svi-lr 0.01 &
poetry run python scripts/wn_network.py --top-n 3 --days 2 --start-day 8 --svi-steps 2000 --plot-every 200 --svi-lr 0.01 &

wait;

# Train on disrupted data
poetry run python scripts/wn_network.py --top-n 3 --days 2 --start-day 0 --svi-steps 2000 --plot-every 200 --svi-lr 0.01 --disrupted &
poetry run python scripts/wn_network.py --top-n 3 --days 2 --start-day 2 --svi-steps 2000 --plot-every 200 --svi-lr 0.01 --disrupted &
poetry run python scripts/wn_network.py --top-n 3 --days 2 --start-day 4 --svi-steps 2000 --plot-every 200 --svi-lr 0.01 --disrupted &
poetry run python scripts/wn_network.py --top-n 3 --days 2 --start-day 8 --svi-steps 2000 --plot-every 200 --svi-lr 0.01 --disrupted &
