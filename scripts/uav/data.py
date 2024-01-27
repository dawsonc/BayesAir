import os

import numpy as np
import pandas as pd
import torch


def load_data(base_path, experiment_path, fields, dt=0.25):
    # Load all the dfs into a list, remapping columns to the names in field_names
    dfs = []
    for field_name, field_map in fields.items():
        path = os.path.join(
            base_path, experiment_path, experiment_path + "-" + field_name + ".csv"
        )
        df = pd.read_csv(path)
        df.rename(columns=field_map, inplace=True)
        dfs.append(df)

    # Get the min and max times
    min_time = min([df["%time"].min() for df in dfs]) * 1e-9
    max_time = max([df["%time"].max() for df in dfs]) * 1e-9

    # Normalize and resample time
    t = np.arange(0, max_time - min_time, dt)
    normalized_dfs = []
    for df, field_map in zip(dfs, fields.values()):
        sampled_times = df["%time"] * 1e-9 - min_time
        normalized_df = pd.DataFrame(index=t, columns=field_map.values())
        normalized_df.index.name = "Time (s)"

        for field in field_map.values():
            # We have to treat the error status specially, since it's only reported
            # when a failure is occuring (and is implicitly zero otherwise)
            if "status" in field:
                normalized_df[field] = np.interp(t, sampled_times, df[field], left=0)
            else:
                normalized_df[field] = np.interp(t, sampled_times, df[field])

            # Handle angles to unwrap them
            if "roll" in field or "pitch" in field or "yaw" in field:
                normalized_df[field] = np.unwrap(normalized_df[field], period=360)

        normalized_dfs.append(normalized_df)

    # Merge all the dataframes into one
    df = pd.concat(normalized_dfs, axis=1, join="inner")

    return df


def df_to_tensors(df, device):
    t = torch.tensor(df.index.to_numpy(), device=device)

    roll = df["roll_measured"].to_numpy() * np.pi / 180
    pitch = df["pitch_measured"].to_numpy() * np.pi / 180
    yaw = df["yaw_measured"].to_numpy() * np.pi / 180

    roll_desired = df["roll_commanded"].to_numpy() * np.pi / 180
    pitch_desired = df["pitch_commanded"].to_numpy() * np.pi / 180
    yaw_desired = df["yaw_commanded"].to_numpy() * np.pi / 180

    p = df["twist_wx"].to_numpy()
    q = -df["twist_wy"].to_numpy()
    r = -df["twist_wz"].to_numpy()

    states = torch.tensor(
        np.stack([roll, pitch, yaw], axis=1),
        dtype=torch.float32,
        device=device,
    )
    initial_states = states[:-1]
    next_states = states[1:]

    pqrs = torch.tensor(
        np.stack([p, q, r], axis=1),
        dtype=torch.float32,
        device=device,
    )
    pqrs = pqrs[:-1]

    desired_states = torch.tensor(
        np.stack([roll_desired, pitch_desired, yaw_desired], axis=1),
        dtype=torch.float32,
        device=device,
    )
    desired_states = desired_states[:-1]

    return t, initial_states, next_states, pqrs, desired_states


def load_all_data(dt=0.25, device=None):
    """Load UAV failure data.

    Returns:
        - List of tensors containing the time, initial and next states, angular
            velocities, and desired states for the each nominal experiment.
        - Same for the elevator failure experiments.
        - Same for the rudder failure experiments.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.join(base_path, "data")
    elevator_failure_experiments = [
        "carbonZ_2018-09-11-14-41-51_elevator_failure",
        "carbonZ_2018-09-11-15-05-11_1_elevator_failure",
    ]
    rudder_failure_experiments = [
        "carbonZ_2018-09-11-15-06-34_1_rudder_right_failure",
        "carbonZ_2018-09-11-15-06-34_2_rudder_right_failure",
        # "carbonZ_2018-09-11-15-06-34_3_rudder_left_failure",
    ]
    nominal_experiments = [
        "carbonZ_2018-07-18-16-37-39_1_no_failure",
        "carbonZ_2018-07-30-16-39-00_3_no_failure",
        "carbonZ_2018-09-11-14-16-55_no_failure",
        "carbonZ_2018-09-11-14-41-38_no_failure",
        "carbonZ_2018-09-11-15-05-11_2_no_failure",
        "carbonZ_2018-10-05-14-34-20_1_no_failure",
        "carbonZ_2018-10-05-14-37-22_1_no_failure",
        "carbonZ_2018-10-05-15-52-12_1_no_failure",
        "carbonZ_2018-10-05-15-52-12_2_no_failure",
        "carbonZ_2018-10-18-11-08-24_no_failure",
    ]
    fields = {
        "mavros-local_position-velocity": {
            "field.twist.linear.x": "twist_vx",
            "field.twist.linear.y": "twist_vy",
            "field.twist.linear.z": "twist_vz",
            "field.twist.angular.x": "twist_wx",
            "field.twist.angular.y": "twist_wy",
            "field.twist.angular.z": "twist_wz",
        },
        "mavros-nav_info-pitch": {
            "field.commanded": "pitch_commanded",
            "field.measured": "pitch_measured",
        },
        "mavros-nav_info-roll": {
            "field.commanded": "roll_commanded",
            "field.measured": "roll_measured",
        },
        "mavros-nav_info-yaw": {
            "field.commanded": "yaw_commanded",
            "field.measured": "yaw_measured",
        },
    }

    # Load the data
    nominal_dfs = [
        load_data(os.path.join(base_path, "nominal"), experiment_path, fields, dt=0.25)
        for experiment_path in nominal_experiments
    ]
    elevator_failure_dfs = [
        load_data(
            os.path.join(base_path, "failure"),
            experiment_path,
            fields | {"failure_status-elevator": {"field.data": "elevator_status"}},
            dt=0.25,
        )
        for experiment_path in elevator_failure_experiments
    ]
    rudder_failure_dfs = [
        load_data(
            os.path.join(base_path, "failure"),
            experiment_path,
            fields | {"failure_status-rudder": {"field.data": "rudder_status"}},
            dt=0.25,
        )
        for experiment_path in rudder_failure_experiments
    ]

    # Trim failure data to the period when a failure is occuring
    for df in elevator_failure_dfs:
        df.drop(df[df["elevator_status"] == 0].index, inplace=True)

    for df in rudder_failure_dfs:
        df.drop(df[df["rudder_status"] == 0].index, inplace=True)

    # Convert the list of DFs into couple of lists of tensors
    nominal_data = [df_to_tensors(df, device) for df in nominal_dfs]
    nominal_data = tuple(map(list, zip(*nominal_data)))
    (
        t_nominal,
        nominal_initial_states,
        nominal_next_states,
        nominal_pqrs,
        nominal_commands,
    ) = nominal_data

    elevator_failure_data = [df_to_tensors(df, device) for df in elevator_failure_dfs]
    elevator_failure_data = tuple(map(list, zip(*elevator_failure_data)))
    (
        t_elevator_failure,
        elevator_failure_initial_states,
        elevator_failure_next_states,
        elevator_failure_pqrs,
        elevator_failure_commands,
    ) = elevator_failure_data

    rudder_failure_data = [df_to_tensors(df, device) for df in rudder_failure_dfs]
    rudder_failure_data = tuple(map(list, zip(*rudder_failure_data)))
    (
        t_rudder_failure,
        rudder_failure_initial_states,
        rudder_failure_next_states,
        rudder_failure_pqrs,
        rudder_failure_commands,
    ) = rudder_failure_data

    return (
        (
            t_nominal,
            nominal_initial_states,
            nominal_next_states,
            nominal_pqrs,
            nominal_commands,
        ),
        (
            t_elevator_failure,
            elevator_failure_initial_states,
            elevator_failure_next_states,
            elevator_failure_pqrs,
            elevator_failure_commands,
        ),
        (
            t_rudder_failure,
            rudder_failure_initial_states,
            rudder_failure_next_states,
            rudder_failure_pqrs,
            rudder_failure_commands,
        ),
    )


if __name__ == "__main__":
    # Load the data and print a summary of how much data we have
    nominal_data, elevator_data, rudder_data = load_all_data()

    print("Nominal Data")
    print("============")
    print("     Number of trials:", len(nominal_data[0]))
    print("Total # of time steps:", sum([len(t) for t in nominal_data[0]]))
    print(" Time steps per trial:", [len(t) for t in nominal_data[0]])
    print()

    print("Elevator Failure Data")
    print("=====================")
    print("     Number of trials:", len(elevator_data[0]))
    print("Total # of time steps:", sum([len(t) for t in elevator_data[0]]))
    print(" Time steps per trial:", [len(t) for t in elevator_data[0]])
    print()

    print("Rudder Failure Data")
    print("===================")
    print("     Number of trials:", len(rudder_data[0]))
    print("Total # of time steps:", sum([len(t) for t in rudder_data[0]]))
    print(" Time steps per trial:", [len(t) for t in rudder_data[0]])

    # # Plot the nominal data
    # trial = 0
    # t_nominal, q_nominal = nominal_data[0], nominal_data[1]
    # plt.plot(t_nominal[trial].cpu()[:-1], q_nominal[trial].cpu())
    # plt.title("Nominal Data")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Angle (rad)")
    # plt.legend(["Roll", "Pitch", "Yaw"])
    # plt.show()

    # # Plot the elevator failure data
    # for trial in range(len(elevator_data[0])):
    #     t_elevator, q_elevator = elevator_data[0], elevator_data[1]
    #     plt.plot(t_elevator[trial].cpu()[:-1], q_elevator[trial].cpu())
    #     plt.title("Elevator Failure Data")
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Angle (rad)")
    #     plt.legend(["Roll", "Pitch", "Yaw"])
    #     plt.show()

    # # Plot the rudder failure data
    # for trial in range(len(rudder_data[0])):
    #     t_rudder, q_rudder = rudder_data[0], rudder_data[1]
    #     plt.plot(t_rudder[trial].cpu()[:-1], q_rudder[trial].cpu())
    #     plt.title("Rudder Failure Data")
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Angle (rad)")
    #     plt.legend(["Roll", "Pitch", "Yaw"])
    #     plt.show()
