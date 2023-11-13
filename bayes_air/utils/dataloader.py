"""Utilities for loading data."""
import os

import pandas as pd


def load_all_data():
    """Get all available WN flight data"""
    # Get the directory of the current file
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the relative path to the CSV file
    file_path = os.path.join(script_directory, "../..", "data", "wn_dec21_dec30.csv")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    return df


def split_nominal_disrupted_data(df: pd.DataFrame):
    """Split dataset into nominal data and disrupted data.

    The disruption occurred between 2022-12-21 and 2023-1-1

    Args:
        df: the dataframe of flight data

    Returns:
        A dataframe filtered to include only flights outside the disrupted period
        A dataframe filtered to include flights within the disrupted period
    """
    # Convert to date type
    df["Date"] = pd.to_datetime(df["Date"])

    # Filter rows based on the date condition
    disrupted_start = pd.to_datetime("12/21/2022")
    disrupted_end = pd.to_datetime("12/30/2022")

    # Filter rows based on the date condition
    nominal_data = df[(df["Date"] < disrupted_start) | (df["Date"] > disrupted_end)]
    disrupted_data = df[(df["Date"] >= disrupted_start) & (df["Date"] <= disrupted_end)]

    return nominal_data, disrupted_data


def split_by_date(df: pd.DataFrame):
    """Split a DataFrame of flights into a list of DataFrames, one for each date.

    Args:
        df: the dataframe of flight data with a 'Date' column

    Returns:
        A list of DataFrames, each containing data for a specific date
    """
    # Group the DataFrame by the 'Date' column
    grouped_df = df.groupby("Date")

    # Create a list of DataFrames, one for each date
    date_dataframes = [group for _, group in grouped_df]

    return date_dataframes


def convert_to_float_hours_optimized(time_series):
    """Convert time in 24-hour format to float hours since midnight.

    Args:
        time_series: a pandas Series representing time in 24-hour format (HH:MM)

    Returns:
        Float hours since midnight, or None for canceled flights
    """
    # Replace "--:--" with "23:59" (delay cancelled flights to end of day)
    time_series.replace("--:--", "23:59", inplace=True)

    # Convert time strings to datetime objects
    time_objects = pd.to_datetime(time_series, format="%H:%M")

    # Extract hour and minute components
    hours_since_midnight = time_objects.dt.hour + time_objects.dt.minute / 60.0

    return hours_since_midnight


def remap_columns(df):
    """Remap columns in the DataFrame to the names that we expect.

    Args:
        df: the original dataframe

    Returns:
        A new dataframe with remapped columns
    """
    # Define the mapping
    column_mapping = {
        "Flight Number": "flight_number",
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

    # Convert all times to hours since midnight
    remapped_df["scheduled_departure_time"] = convert_to_float_hours_optimized(
        remapped_df["scheduled_departure_time"]
    )
    remapped_df["scheduled_arrival_time"] = convert_to_float_hours_optimized(
        remapped_df["scheduled_arrival_time"]
    )
    remapped_df["actual_departure_time"] = convert_to_float_hours_optimized(
        remapped_df["actual_departure_time"]
    )
    remapped_df["actual_arrival_time"] = convert_to_float_hours_optimized(
        remapped_df["actual_arrival_time"]
    )

    return remapped_df

def top_N_df(df, number_of_airports):
    number_of_airports = number_of_airports

    #Using df.mode() to determine most common elements in a column.
    mode = df.mode()

    #Creating a list of "top airports" by destination_airport. 
    most_visited_airports = []
    for i in range(number_of_airports):
        most_visited_airports.append(mode["desintation_airport"][i])

    #Only choose flights that have flights between two airports in that list
    for i in range(len(df.index)):

        #Only appending the flight if its airport is in the most_visited_airports_list
        A1 = df["origin_airport"][i]
        A2 = df["destination_airport"][i]
        if(A1 not in most_visited_airports or A2 not in most_visited_airports):
            df = df.drop([i])
    
    #Resetting the indices of the datagrame
    df_reset = df.reset_index(drop=True)

    return df_reset

if __name__ == "__main__":
    df = load_all_data()
    nominal_df, disrupted_df = split_nominal_disrupted_data(df)
    nominal_dfs, disrupted_dfs = split_by_date(nominal_df), split_by_date(disrupted_df)
    nominal_dfs = [remap_columns(df) for df in nominal_dfs]
    disrupted_dfs = [remap_columns(df) for df in disrupted_dfs]
