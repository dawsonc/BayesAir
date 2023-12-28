"""Utilities for loading data."""
import os

import pandas as pd
import timezonefinder as tzf


def load_all_data():
    """Get all available WN flight data"""
    # Get the directory of the current file
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the relative path to the CSV file
    nominal_file_path = os.path.join(
        script_directory, "../..", "data", "wn_dec01_dec20.csv"
    )
    disrupted_file_path = os.path.join(
        script_directory, "../..", "data", "wn_dec21_dec30.csv"
    )
    airport_locations_file_path = os.path.join(
        script_directory, "../..", "data", "airport_locations.csv"
    )

    # Read the CSV files into a DataFrame
    nominal_df = pd.read_csv(nominal_file_path)
    disrupted_df = pd.read_csv(disrupted_file_path)
    airport_locations_df = pd.read_csv(airport_locations_file_path)

    # Filter airport locations to only keep the latest
    airport_locations_df = airport_locations_df[
        airport_locations_df.AIRPORT_IS_LATEST.astype(bool)
    ]

    # Concatenate the two DataFrames
    df = pd.concat([nominal_df, disrupted_df])

    return df, airport_locations_df


def split_nominal_disrupted_data(df: pd.DataFrame):
    """Split dataset into nominal data and disrupted data.

    The disruption occurred between 2022-12-21 and 2023-1-1

    Args:
        df: the dataframe of flight data

    Returns:
        A dataframe filtered to include only flights outside the disrupted period
        A dataframe filtered to include flights within the disrupted period
    """
    # Filter rows based on the date condition
    disrupted_start = pd.to_datetime("12/21/2022")
    disrupted_end = pd.to_datetime("12/30/2022")

    # Filter rows based on the date condition
    nominal_data = df[(df["date"] < disrupted_start) | (df["date"] > disrupted_end)]
    disrupted_data = df[(df["date"] >= disrupted_start) & (df["date"] <= disrupted_end)]

    return nominal_data, disrupted_data


def split_by_date(df: pd.DataFrame):
    """Split a DataFrame of flights into a list of DataFrames, one for each date.

    Args:
        df: the dataframe of flight data with a "date" column

    Returns:
        A list of DataFrames, each containing data for a specific date
    """
    # Group the DataFrame by the "date" column
    grouped_df = df.groupby("date")

    # Create a list of DataFrames, one for each date
    date_dataframes = [group for _, group in grouped_df]

    # Sort within each date by scheduled departure time
    for date_df in date_dataframes:
        date_df.sort_values(by="scheduled_departure_time", inplace=True)

    return date_dataframes


def convert_to_float_hours_optimized(time_series, time_zone_series):
    """Convert time in 24-hour format to float hours since midnight.

    Args:
        time_series: a pandas Series representing time in 24-hour format (HH:MM)
        time_zone_series: a pandas Series representing the time zone of each time

    Returns:
        Float hours since midnight, or None for canceled flights
    """
    # Replace "--:--" with "00:00" (mark them for later modification)
    time_series.replace("--:--", "00:00", inplace=True)

    # Replace "24:00" with "23:59" (midnight)
    time_series.replace("24:00", "23:59", inplace=True)

    # Convert time strings to datetime objects
    time_objects = pd.to_datetime(time_series, format="%H:%M")

    # Convert time objects to UTC
    combined_df = pd.concat([time_objects, time_zone_series], axis=1)
    time_objects = combined_df.apply(
        lambda row: row.iloc[0].tz_localize(row.iloc[1]).tz_convert("UTC"), axis=1
    )

    # Extract hour and minute components
    hours_since_midnight = time_objects.dt.hour + time_objects.dt.minute / 60.0

    # Replace times for cancelled flights with the maximum observed time + 1
    hours_since_midnight[time_series == "00:00"] = hours_since_midnight.max() + 1.0

    return hours_since_midnight


def time_zone_for_airports(airport_codes, airport_locations_df):
    """Get the time zone for each airport.

    Args:
        airport_codes: a list of airport codes
        airport_locations_df: a dataframe containing airport locations

    Returns:
        A list of time zones corresponding to each airport
    """
    # Get the time zone for each airport
    finder = tzf.TimezoneFinder()
    time_zones = []
    for airport_code in airport_codes:
        latitude = airport_locations_df[
            airport_locations_df.AIRPORT == airport_code
        ].LATITUDE.iat[0]
        longitude = airport_locations_df[
            airport_locations_df.AIRPORT == airport_code
        ].LONGITUDE.iat[0]
        time_zones.append(finder.timezone_at(lng=longitude, lat=latitude))

    return time_zones


def remap_columns(df, airport_locations_df):
    """Remap columns in the DataFrame to the names that we expect.

    Args:
        df: the original dataframe
        airport_locations_df: a dataframe containing airport locations

    Returns:
        A new dataframe with remapped columns
    """
    # Define the mapping
    column_mapping = {
        "Flight Number": "flight_number",
        "Date": "date",
        "Origin Airport Code": "origin_airport",
        "Dest Airport Code": "destination_airport",
        "Scheduled Departure Time": "scheduled_departure_time",
        "Scheduled Arrival Time": "scheduled_arrival_time",
        "Actual Departure Time": "actual_departure_time",
        "Actual Arrival Time": "actual_arrival_time",
    }

    # Filter the original DataFrame based on the desired columns
    remapped_df = df[column_mapping.keys()]

    # Rename the columns based on the mapping
    remapped_df = remapped_df.rename(columns=column_mapping)

    # Get a list of airport time zones
    airport_codes = pd.concat(
        [
            remapped_df["origin_airport"],
            remapped_df["destination_airport"],
        ]
    ).unique()
    airport_time_zones = pd.DataFrame(
        {
            "airport_code": airport_codes,
            "time_zone": time_zone_for_airports(airport_codes, airport_locations_df),
        }
    )

    # Add a column for time zones to remapped_df using a merge
    remapped_df = remapped_df.merge(
        airport_time_zones,
        left_on="origin_airport",
        right_on="airport_code",
    )
    remapped_df = remapped_df.rename(columns={"time_zone": "origin_time_zone"})
    remapped_df = remapped_df.merge(
        airport_time_zones,
        left_on="destination_airport",
        right_on="airport_code",
    )
    remapped_df = remapped_df.rename(columns={"time_zone": "destination_time_zone"})

    # Convert all times to hours since midnight
    remapped_df["scheduled_departure_time"] = convert_to_float_hours_optimized(
        remapped_df["scheduled_departure_time"],
        remapped_df["origin_time_zone"],
    )
    remapped_df["scheduled_arrival_time"] = convert_to_float_hours_optimized(
        remapped_df["scheduled_arrival_time"],
        remapped_df["destination_time_zone"],
    )
    remapped_df["actual_departure_time"] = convert_to_float_hours_optimized(
        remapped_df["actual_departure_time"],
        remapped_df["origin_time_zone"],
    )
    remapped_df["actual_arrival_time"] = convert_to_float_hours_optimized(
        remapped_df["actual_arrival_time"],
        remapped_df["destination_time_zone"],
    )

    # If a flight is en-route at midnight, it's duration will be negative unless we add 24 hours
    # to the actual and scheduled arrival times
    scheduled_duration = (
        remapped_df.scheduled_arrival_time - remapped_df.scheduled_departure_time
    )
    actual_duration = (
        remapped_df.actual_arrival_time - remapped_df.actual_departure_time
    )
    remapped_df.loc[
        actual_duration < 0, "actual_arrival_time"
    ] += 24  # Add 24 hours to actual arrival time
    remapped_df.loc[
        scheduled_duration < 0, "scheduled_arrival_time"
    ] += 24  # Add 24 hours to scheduled arrival time

    # If a flight is delayed so that both the actual departure is the next day,
    # we need to add 24 hours to the actual departure and arrival times
    departure_delay = (
        remapped_df.actual_departure_time - remapped_df.scheduled_departure_time
    )
    # Departure delay will be positive for flights that depart late within the same day,
    # slightly negative for flights that depart early within the same day, and very
    # negative for flights that depart late the next day
    remapped_df.loc[
        departure_delay < -3.0,
        "actual_departure_time",
    ] += 24  # Add 24 hours to actual departure time
    remapped_df.loc[
        departure_delay < -3.0,
        "actual_arrival_time",
    ] += 24  # Add 24 hours to actual arrival time

    # Convert date to datetime type
    remapped_df["date"] = pd.to_datetime(remapped_df["date"])

    return remapped_df


def top_N_df(df, number_of_airports: int):
    """
    Get the top N airports by arrivals and filter the dataframe to include only
    flights between those airports.

    Args:
        df: the original dataframe
        number_of_airports: the number of airports to include
    """
    # Get the top-N airports by arrivals
    top_N_airports = (
        df["destination_airport"].value_counts().head(number_of_airports).index
    )

    # Filter the original DataFrame based on the desired airports
    filtered_df = df[
        df["origin_airport"].isin(top_N_airports)
        & df["destination_airport"].isin(top_N_airports)
    ]

    return filtered_df


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load data, filter, and split by date
    df, airport_locations_df = load_all_data()
    df = remap_columns(df, airport_locations_df)
    filtered_df = top_N_df(df, 6)
    nominal_df, disrupted_df = split_nominal_disrupted_data(filtered_df)
    nominal_dfs, disrupted_dfs = split_by_date(nominal_df), split_by_date(disrupted_df)

    # Save remapped data to file
    script_directory = os.path.dirname(os.path.abspath(__file__))
    df.to_pickle(os.path.join(script_directory, "../..", "data", "wn_data_clean.pkl"))

    # Plot a histogram of the total number of flights between top-N airports
    N_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_flights = []
    for top_N in N_range:
        filtered_df = top_N_df(df, top_N)
        num_flights.append(len(filtered_df))

    plt.plot(N_range, num_flights, "o-")
    plt.xlabel("Number of airports kept in dataset")
    plt.ylabel("Total number of flights")
    plt.show()
