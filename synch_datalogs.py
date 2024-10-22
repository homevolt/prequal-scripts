#!/usr/bin/env python

import os
import pandas as pd
import re
import argparse
from datetime import datetime

# datalog_10.241.64.200_50066_2024-03-26-15_00_16.csv
fn_re = re.compile(
    r"datalog_(\d+\.\d+\.\d+\.\d+)_(\d+)_(\d{4}-\d{2}-\d{2}-\d{2}_\d{2}_\d{2}).csv"
)

# datalog_10.241.64.200_50066_2024-03-26-15_00_16_metadata.csv
fn_metadata_re = re.compile(
    r"datalog_(\d+\.\d+\.\d+\.\d+)_(\d+)_(\d{4}-\d{2}-\d{2}-\d{2}_\d{2}_\d{2})_metadata.csv"
)

temp_dir_path = None


def split_and_check_filenames(filenames):
    m_fns = []
    d_fns = []

    for f in filenames:
        m = fn_re.match(os.path.basename(f.name))
        if m:
            d_fns.append(f)
            continue

        m = fn_metadata_re.match(os.path.basename(f.name))
        if m:
            m_fns.append(f)
            continue

        raise ValueError(f"File {f.name} does not match the expected format")

    return (d_fns, m_fns)


def parse_files(filenames):
    "Concat, sort, round and drop duplicates from all given files"
    frames = []
    interval = None

    for f in filenames:
        m = fn_re.match(os.path.basename(f.name))
        if not m:
            raise ValueError(f"File {f.name} does not match the expected format")

        print(f"Parsing {f.name}...")

        df = pd.read_csv(f, parse_dates=[0], date_format="%Y-%m-%d %H:%M:%S.%f%z")
        # Add the IP address so that duplicated rows from same IP can be removed.
        # This can happen if the datalog server is restarted or a network issue and the same data is sent again.
        df["ip"] = m.group(1)

        if df.empty:
            print(f"File {f.name} is empty, skipping...")
            continue

        delta_s = (df["timestamp"][1] - df["timestamp"][0]).total_seconds()

        if interval == None:
            interval = delta_s
        elif interval != delta_s:
            raise ValueError(
                f"File {f.name} doesn't have same interval as the previous files"
            )

        dups = df[df["timestamp"].duplicated(keep=False)]
        if not dups.empty:
            print(f"File {f.name} has duplicated timestamps")
            print(dups)
            raise ValueError(f"File {f.name} has duplicates")

        # Ensure that the timestamps are consecutive and unique
        ts_counts = df["timestamp"].diff().value_counts()
        if len(ts_counts) != 1:
            raise ValueError(
                f"File {f.name} has more than one interval:\n{ts_counts.to_string()}"
            )

        frames.append(df)
    df = pd.concat(frames, ignore_index=True).sort_values(by="timestamp")

    print(f"Rounding to {interval} seconds...")
    df["timestamp"] = df["timestamp"].dt.floor(f"{interval}s")

    print("Dropping duplicates...")
    df.drop_duplicates(inplace=True, ignore_index=True)

    # Save a CSV file for each unique IP address
    unique_ips = df["ip"].unique()
    for ip in unique_ips:
        ip_df = df[df["ip"] == ip]
        ip_df["timestamp"] = ip_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S.%f%z")
        ip_filename = os.path.join(temp_dir_path, f"combined_datalog_{ip}.csv")
        print(f"Saving combined data for IP {ip}")
        ip_df.to_csv(ip_filename, index=False)

    return df


def parse_metadata_files(filenames):
    "Concat, sort and drop duplicates from all given files"
    frames = []

    for f in filenames:
        m = fn_metadata_re.match(os.path.basename(f.name))
        if not m:
            raise ValueError(f"File {f.name} does not match the expected format")

        print(f"Parsing {f.name}...")

        df = pd.read_csv(f, parse_dates=[0], date_format="%Y-%m-%d %H:%M:%S.%f%z")
        # Add the IP address so that duplicated rows from same IP can be removed.
        # This can happen if the datalog server is restarted or a network issue and the same data is sent again.
        df["ip"] = m.group(1)

        if df.empty:
            print(f"File {f.name} is empty, skipping...")
            continue

        dups = df[df["timestamp"].duplicated(keep=False)]
        if not dups.empty:
            print(f"File {f.name} has duplicated timestamps")
            print(dups)
            raise ValueError(f"File {f.name} has duplicates")

        frames.append(df)
    df = pd.concat(frames, ignore_index=True).sort_values(by="timestamp")

    delta_s = 0.1
    print(f"Rounding to {delta_s} seconds...")
    df["timestamp"] = df["timestamp"].dt.floor(f"{delta_s}s")

    print("Dropping duplicates...")
    df.drop_duplicates(inplace=True, ignore_index=True)

    # Save a CSV file for each unique IP address
    unique_ips = df["ip"].unique()
    for ip in unique_ips:
        ip_df = df[df["ip"] == ip]
        ip_df["timestamp"] = ip_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S.%f%z")
        ip_filename = os.path.join(temp_dir_path, f"combined_datalog_{ip}_metadata.csv")
        print(f"Saving combined metadata for IP {ip}...")
        ip_df.to_csv(ip_filename, index=False)

    return df


def adjust_datalogs_and_metadata(folder_path, args):
    metadata_pattern = re.compile(r"combined_datalog_(.*)_metadata\.csv")
    datalog_pattern = re.compile(r"combined_datalog_(.*)\.csv")

    # List all files in the directory
    all_files = os.listdir(folder_path)
    metadata_files = [f for f in all_files if metadata_pattern.match(f)]
    datalog_files = {
        m.group(1): os.path.join(folder_path, f)
        for f in all_files
        if (m := datalog_pattern.match(f))
    }

    if not metadata_files:
        raise ValueError("No metadata files found in the specified directory")

    # Read the first metadata file to use its timestamps as the reference
    first_metadata_file = os.path.join(folder_path, metadata_files[0])
    first_metadata_df = pd.read_csv(
        first_metadata_file, parse_dates=[0], date_format="%Y-%m-%d %H:%M:%S.%f%z"
    )
    reference_timestamps = first_metadata_df["timestamp"]
    reference_period_values = first_metadata_df["fcr_test_sine_period_milli_second"]
    num_rows = len(reference_timestamps)

    # Check and adjust the other metadata files
    for metadata_file in metadata_files:
        metadata_file_path = os.path.join(folder_path, metadata_file)
        metadata_df = pd.read_csv(metadata_file_path, parse_dates=["timestamp"])

        if args.down:
            if len(metadata_df) != num_rows:
                raise ValueError(
                    f"File {metadata_file} does not have the same number of rows as the first metadata file"
                )

            if not metadata_df["fcr_test_sine_period_milli_second"].equals(
                reference_period_values
            ):
                raise ValueError(
                    f"File {metadata_file} does not have the same values in 'fcr_test_sine_period_milli_second' as the first metadata file"
                )

        ip = metadata_pattern.match(metadata_file).group(1)
        if ip not in datalog_files:
            raise ValueError(f"Corresponding datalog file for IP {ip} not found")

        datalog_file_path = datalog_files[ip]
        datalog_df = pd.read_csv(datalog_file_path, parse_dates=["timestamp"])

        # Calculate the interval between datalog timestamps
        datalog_interval = (
            datalog_df["timestamp"][1] - datalog_df["timestamp"][0]
        ).total_seconds()

        # Trim the datalog file to keep timestamps within the range of the metadata timestamps
        metadata_first_timestamp = metadata_df["timestamp"].iloc[0]
        metadata_last_timestamp = metadata_df["timestamp"].iloc[-1]
        trimmed_datalog_df = datalog_df.loc[
            (datalog_df["timestamp"] >= metadata_first_timestamp)
            & (datalog_df["timestamp"] <= metadata_last_timestamp)
        ].copy()

        if trimmed_datalog_df.empty:
            print(f"No valid timestamps found in {datalog_file_path}")
            continue

        # Replace the first timestamp by the reference
        trimmed_datalog_df.loc[trimmed_datalog_df.index[0], "timestamp"] = (
            reference_timestamps[0]
        )

        # Replace all following datalog timestamps by stepping up the calculated interval
        for i in range(1, len(trimmed_datalog_df)):
            trimmed_datalog_df.loc[trimmed_datalog_df.index[i], "timestamp"] = (
                trimmed_datalog_df["timestamp"].iloc[i - 1]
                + pd.Timedelta(seconds=datalog_interval)
            )

        # Save the adjusted datalog file
        trimmed_datalog_df["timestamp"] = trimmed_datalog_df["timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S.%f%z"
        )

        trimmed_datalog_df.to_csv(datalog_file_path, index=False)
        print(f"Adjusted datalog timestamps for IP: {ip}")

        # Replace the timestamps in the metadata file with the reference timestamps
        metadata_df["timestamp"] = reference_timestamps.dt.strftime(
            "%Y-%m-%d %H:%M:%S.%f%z"
        )
        metadata_df.drop_duplicates(
            subset=["timestamp"], inplace=True, ignore_index=True
        )

        # Save the adjusted metadata file
        metadata_df.to_csv(metadata_file_path, index=False)
        print(f"Adjusted metadata timestamps for IP {ip}")


def adjust_datalogs(folder_path):
    datalog_pattern = re.compile(r"combined_datalog_(.*)\.csv")

    # List all files in the directory
    all_files = os.listdir(folder_path)
    datalog_files = {
        m.group(0): os.path.join(folder_path, f)
        for f in all_files
        if (m := datalog_pattern.match(f))
    }

    if not datalog_files:
        raise ValueError("No datalog files found in the specified folder")

    # Read the first datalog file to use its timestamps as the reference
    first_datalog_key = next(
        iter(datalog_files)
    )  # Get the first key from the dictionary
    first_datalog_file = datalog_files[first_datalog_key]
    first_datalog_df = pd.read_csv(
        first_datalog_file, parse_dates=[0], date_format="%Y-%m-%d %H:%M:%S.%f%z"
    )

    reference_timestamps = first_datalog_df["timestamp"]

    # Check and adjust the other metadata files
    for datalog_key, datalog_file_path in datalog_files.items():
        datalog_df = pd.read_csv(datalog_file_path, parse_dates=["timestamp"])

        # Calculate the interval between datalog timestamps
        datalog_interval = (
            datalog_df["timestamp"][1] - datalog_df["timestamp"][0]
        ).total_seconds()

        # Replace the first timestamp by the reference
        datalog_df.loc[datalog_df.index[0], "timestamp"] = reference_timestamps.iloc[0]

        # Replace all following datalog timestamps by stepping up the calculated interval
        for i in range(1, len(datalog_df)):
            datalog_df.loc[datalog_df.index[i], "timestamp"] = datalog_df[
                "timestamp"
            ].iloc[i - 1] + pd.Timedelta(seconds=datalog_interval)

        # Save the adjusted datalog file
        datalog_df["timestamp"] = datalog_df["timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S.%f%z"
        )

        datalog_df.to_csv(datalog_file_path, index=False)
        ip = datalog_pattern.match(os.path.basename(datalog_file_path)).group(1)
        print(f"Adjusted datalog timestamps for IP: {ip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        type=argparse.FileType("r"),
        help="File(s) from datalog server, - for stdin.",
        nargs="+",
    )
    parser.add_argument(
        "-d",
        "--down",
        action="store_true",
        help="FCR-D ramp down test, metadata consistency will be checked",
    )
    parser.add_argument(
        "-nm",
        "--no_metadata",
        action="store_true",
        help="Pre-process datalogs without metadata",
    )
    args = parser.parse_args()

    # Avoiding warning, ref:
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    pd.options.mode.copy_on_write = True

    # Get the directory of the first file passed as an argument
    first_file_path = args.filename[0].name
    base_dir = os.path.dirname(first_file_path)

    # Create a temporary directory in the same folder as the files
    temp_dir_path = os.path.join(base_dir, "temp_data")
    os.makedirs(temp_dir_path, exist_ok=True)

    d_fns, m_fns = split_and_check_filenames(args.filename)
    df = parse_files(d_fns)

    if args.no_metadata:
        adjust_datalogs(temp_dir_path)
    else:
        m_df = parse_metadata_files(m_fns)
        adjust_datalogs_and_metadata(temp_dir_path, args)
