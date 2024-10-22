#!/usr/bin/env python

import os
import pandas as pd
import re
import argparse
from datetime import datetime

# datalog_10.241.64.200.csv
fn_re = re.compile(r"combined_datalog_(\d+\.\d+\.\d+\.\d+).csv")

# datalog_10.241.64.200_metadata.csv
fn_metadata_re = re.compile(
    r"combined_datalog_(\d+\.\d+\.\d+\.\d+)_metadata.csv"
)


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


def parse_files(filenames, args):
    "Concat, sort, round and drop duplicates from all given files"
    frames=[]
    interval=None

    for f in filenames:
        m = fn_re.match(os.path.basename(f.name))
        if not m:
            raise ValueError(f"File {f.name} does not match the expected format")

        print(f"Parsing {f.name}...")

        df = pd.read_csv(f, parse_dates=[0], date_format="%Y-%m-%d %H:%M:%S.%f%z")

        if args.ler:
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
            print(f"previous interval={interval}\tcurrent interval={delta_s}")
            raise ValueError(f"File {f.name} doesn't have same interval as the previous files")

        dups = df[df["timestamp"].duplicated(keep=False)]
        if not dups.empty:
            print(f"File {f.name} has duplicated timestamps")
            print(dups)
            raise ValueError(f"File {f.name} has duplicates")

        # Ensure that the timestamps are consecutive and unique
        ts_counts = df["timestamp"].diff().value_counts()
        if len(ts_counts) != 1:
            raise ValueError(f"File {f.name} has more than one interval:\n{ts_counts.to_string()}")

        frames.append(df)
    df = pd.concat(frames, ignore_index=True).sort_values(by="timestamp")

    if args.ler:
        print(f"Rounding to {interval} seconds...")
        df["timestamp"] = df["timestamp"].dt.floor(f"{interval}s")

        print("Dropping duplicates...")
        df.drop_duplicates(subset=["timestamp"],inplace=True, ignore_index=True)

    return df

def analyze_data(df):
    "Analyze data for sanity checks and print some statistics"

    print("Grouping by timestamp...")
    grouped = df.groupby("timestamp", as_index=False)

    freq_info = grouped["grid_frequency_milli_hertz"].agg(["std", "count", lambda x: x.max() - x.min()])
    print("Frequency info")
    print(f" Number of samples per slot: {freq_info['count'].min()} - {freq_info['count'].max()}")
    print(f" Frequency std deviation (mHz): {freq_info['std'].min().round(1)} - {freq_info['std'].max().round(1)}")
    print(f" Frequency delta (mHz): {freq_info['<lambda_0>'].min()} - {freq_info['<lambda_0>'].max()}")
    print("")
    print("Basic sanity checks")
    power_diff = df["active_power_watt"] - df["fcr_n_power_output_watt"] - df["fcr_d_power_output_watt"] - df["ffr_power_output_watt"] - df["nem_power_watt"]
    print(f" Active power deviation is between {power_diff.min()} W and {power_diff.max()} W")
    bad_df = df[power_diff.abs().gt(60)]
    if not bad_df.empty:
        print("Active power deviation above 60 W")
        print(bad_df)
    else:
        print(" No active power deviation above 60 W")

    # Check if the number of samples per slot is constant except for the beginning and end.
    # Groupby the "count" column to find out how many of each count there is, assume that the
    # maximum count is the expected count and require that to be the maximum number of samples per slot.
    count_values = freq_info.value_counts("count")
    expected_count = count_values.idxmax()

    if freq_info["count"].max() != expected_count:
        raise ValueError(f"Expcted the maximum number of samples per slot to be {expected_count}:\n{count_values.to_string()}")

    # Get indexes of all rows with the expected count, ensure that the length between first and last index matches the size of it.
    idx = freq_info[freq_info["count"] == expected_count].index
    if idx[-1] - idx[0] + 1 != len(idx):
        raise ValueError(f"Expected the number of slots to be constant except from beginning and end:\n{count_values.to_string()}")

    print(f" Number of units found: {expected_count}")
    print("All checks passed")

def convert_to_svk(df):
    aggs = {
        "InsAcPow":("active_power_watt", "sum"),
        "GridFreq":("grid_frequency_milli_hertz", "mean"),
        "ContStatus_Fcrn":("fcr_n_state", lambda x: int(all(x))),
        "ContStatus_FcrdUp":("fcr_d_up_state", lambda x: int(all(x))),
        "ContStatus_FcrdDo":("fcr_d_down_state", lambda x: int(all(x))),
        "Pmin":("p_min_watt", "sum"),
        "Pmax":("p_max_watt", "sum"),
        "Activated_Fcrn":("fcr_n_power_output_watt", "sum"),
        "Activated_FcrdUp":("fcr_d_power_output_watt", "sum"),
        "Activated_FcrdDo":("ffr_power_output_watt", "sum"),
        "ResSize_Fcrn":("endurance_fcr_n_second", "min"),
        "ResSize_FcrdUp":("endurance_fcr_d_up_second", "min"),
        "ResSize_FcrdDo":("endurance_fcr_d_down_second", "min"),
        "NEM":("nem_power_watt", "sum"),
        "AEM":("aem", lambda x: int(any(x))),
    }

    print("Grouping by timestamp and aggregating data...")
    df = df.groupby("timestamp", as_index=False).agg(**{k:v for k,v in aggs.items() if v[0] in df.columns})
    df.rename(columns={"timestamp": "DateTime"}, inplace=True)
    # TODO: Consider reindex if all fields are required
    #.reindex(aggs.keys(), axis=1, fill_value='')

    print("Converting to SVK units...")
    # This will convert to MW, Minutes and DateTime with milliseconds instead of microseconds.
    # Converter all fields to strings so that they are correctly exported in the end.
    for i in ("InsAcPow", "Pmin", "Pmax", "Activated_Fcrn", "Activated_FcrdUp", "Activated_FcrdDo", "NEM"):
        if i in df.columns:
            df[i] = (df[i] / 1e6).round(5).apply(lambda x: format(x, '.5f') if x != 0 else "0")
    for i in ("ResSize_Fcrn", "ResSize_FcrdUp", "ResSize_FcrdDo"):
        if i in df.columns:
            df[i] = (df[i] / 60).round(3).apply(lambda x:  format(x, '.3f') if x != 0 else "0")
    df["GridFreq"] = (df["GridFreq"] / 1e3).round(3).apply(lambda x:  format(x, '.3f') if x != 0 else "0")
    df["DateTime"] = df["DateTime"].dt.strftime('%Y%m%dT%H%M%S.%f').apply(lambda x: x[:-3])

    return df

def parse_metadata_files(filenames, args):
    "Concat, sort and drop duplicates from all given files"
    frames = []

    for f in filenames:
        m = fn_metadata_re.match(os.path.basename(f.name))
        if not m:
            raise ValueError(f"File {f.name} does not match the expected format")

        print(f"Parsing {f.name}...")

        df = pd.read_csv(f, parse_dates=[0], date_format="%Y-%m-%d %H:%M:%S.%f%z")

        if args.ler:
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

    if args.ler:
        delta_s = 0.1
        print(f"Rounding to {delta_s} seconds...")
        df["timestamp"] = df["timestamp"].dt.floor(f"{delta_s}s")

    print("Dropping duplicates...")

    return df

def convert_to_eo_metadata(m_df):
    out = pd.DataFrame()
    # Ensure the timestamp column is of datetime type
    m_df["timestamp"] = pd.to_datetime(m_df["timestamp"]).dt.round("1s")
    # Ensure that numeric columns are of appropriate types
    numeric_columns = [
        'pref_watt', 'fcr_d_up_pmax_watt', 'fcr_d_down_pmax_watt',
        'fcr_n_pmax_watt', 'ffr_pmax_watt', 'fcr_test_sine_period_milli_second',
        'test_signal'
    ]
    m_df[numeric_columns] = m_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Group by timestamp and sum/aggregate the appropriate columns
    grouped_df = m_df.groupby('timestamp', as_index=False).agg({
        "pref_watt": "sum",
        "fcr_d_up_pmax_watt": "sum",
        "fcr_d_down_pmax_watt": "sum",
        "fcr_n_pmax_watt": "sum",
        "ffr_pmax_watt": "sum",
        "fcr_test_sine_period_milli_second": "min",
        "test_signal": "min"
    }).reset_index()

    out = pd.DataFrame()
    out["time"] = grouped_df["timestamp"]
    out["fcr_test_start"] = out["time"].apply(lambda x: int(x.timestamp()))

    out["fcr_test_params"] = grouped_df[
        [
            "pref_watt",  # power
            "fcr_d_up_pmax_watt",  # fcr_d_up
            "fcr_d_down_pmax_watt",  # fcr_d_down
            "fcr_n_pmax_watt",  # fcr_n
            "ffr_pmax_watt",  # ffr
        ]
    ].agg(list, axis="columns")

    # Better solution would be to use the proper names from the protobuf file, but that is just a mess.
    test_signal_names = ("end-of-test", "fcr-n-step", "fcr-n-step-endurance", "fcr-d-up-ramp", "fcr-d-up-ramp-endurance", "fcr-d-down-ramp", "fcr-d-down-ramp-endurance", "fcr-n-sine", "fcr-d-up-sine", "fcr-d-down-sine", "fcr-n-ler", "fcr-d-up-ler", "fcr-d-down-ler")
    out["fcr_test_sequence"] = m_df["test_signal"].apply(lambda x: test_signal_names[x] if x < len(test_signal_names) else f"Unknown test {x}")
    out["fcr_test_sine_period"] = (
        m_df["fcr_test_sine_period_milli_second"] / 1000
    ).astype(int)

    # Drop all rows that aren't of interest.
    return out.drop(out[out["fcr_test_sequence"] == "end-of-test"].index)

def crop_to_metadata(df, m_df):
    "Crop the data to the metadata fields, return a tuple with the cropped data and the LER data frames"
    print("Crop data according to metadata so that only test signals are included")
    frames=[]
    ler_frames=[]
    in_test_frame = False
    start_of_test_idx = None
    for i, r in m_df.iterrows():
        if in_test_frame:
            if not r["test_signal"] or (r["test_signal"] in (7, 8) and r["fcr_test_sine_period_milli_second"] == 0):
                # End of test frame, grab data into frames
                # Consider Sine test with 0 period as end of frame so that the data between sine tests are removed.
                # One could of course remove the test frequency from the test_signals.h in the system instead.
                in_test_frame = False
                frames.append(df[(df["timestamp"] >= m_df.iloc[start_of_test_idx]["timestamp"]) & (df["timestamp"] <= r["timestamp"])])
                print("Found end of test signal at", r["timestamp"], "appending", len(frames[-1]), "rows")

                if m_df.iloc[start_of_test_idx]["test_signal"] in (10, 11, 12):
                    print("...and it was a LER test signal, append it to ler_frames as well.")
                    ler_frames.append((m_df.iloc[start_of_test_idx], frames[-1]))
        else:
            # Currently not in a test frame
            if r["test_signal"]:
                # Start of test frame
                print("Found start of test signal %u period %u ms at %s" % (r["test_signal"], r["fcr_test_sine_period_milli_second"], r["timestamp"]))
                in_test_frame = True
                start_of_test_idx = i
            else:
                # Skip non-test frames
                continue

    if in_test_frame:
        print("Warning: Found start of test signal but no end of test signal")
        frames.append(df[df["DateTime"] >= m_df.iloc[start_of_test_idx]["timestamp"]])

    return (pd.concat(frames), ler_frames)

def get_output_fn(df, svk_df, args, test_type="Operation"):
    services =[]
    for x in ("ContStatus_Fcrn", "ContStatus_FcrdUp", "ContStatus_FcrdDo"):
        if x in svk_df.columns and svk_df[x].any():
            services.append(x.split("_")[1])

    timezone = "UTC"
    start_t = df["timestamp"].min().strftime('%Y%m%dT%H%M')
    end_t = df["timestamp"].max().strftime('%Y%m%dT%H%M')
    sampling_rate = int((df["timestamp"][1] - df["timestamp"][0]).total_seconds()*1000)
    date = datetime.now().strftime('%Y%m%d')

    return f"{args.system_name}_{'-'.join(services)}_{test_type}_{args.area}_{timezone}_{start_t}-{end_t}_{sampling_rate}ms_{date}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=argparse.FileType('r'), help="File(s) from datalog server, - for stdin.", nargs="+")
    parser.add_argument('-n', '--system-name', help='System name (for export file name)', required=True)
    parser.add_argument('-a', '--area', help='Area (Elområde) [SE3]', default='SE3')
    parser.add_argument('-l', '--ler', action='store_true', help='LER test datalogs(no pre script treatment)')
    args = parser.parse_args()

    d_fns, m_fns = split_and_check_filenames(args.filename)
    df = parse_files(d_fns, args)
    analyze_data(df)
    svk_df = convert_to_svk(df)

    m_df = None
    if m_fns:
        #if len(df["ip"].value_counts()) != 1:
        #    raise ValueError("All files must be from the same IP address when metadata is provided")
        m_df = parse_metadata_files(m_fns, args)

    fn = get_output_fn(df, svk_df, args)

    if m_df is not None:
        # There is metadata, change the filename from default to CroppedTest
        fn = get_output_fn(df, svk_df, args, test_type="CroppedTest")

        eo_df = convert_to_eo_metadata(m_df)
        eo_df.to_csv(fn + "_metadata.csv", index=False, lineterminator='\r\n')

        # Re-add some columns to the data frame so that they can be used for cropping.
        # Also, avoids converting from Timestamp -> str -> Timestamp
        add_cols = ["timestamp", "fcr_n_f_ref_milli_hz", "fcr_d_up_f_ref_milli_hz", "fcr_d_down_f_ref_milli_hz"]
        svk_df[add_cols] = df[add_cols]
        (svk_df, lers) = crop_to_metadata(svk_df, m_df)
        svk_df.drop(columns=add_cols, inplace=True)

        for (ler_m_data, ler_df) in lers:
            # Add FCR capacity columns [MW]
            ler_df["Cap_Fcrdn"] = ler_m_data["fcr_n_pmax_watt"] / 1e6
            ler_df["Cap_FcrdUp"] = ler_m_data["fcr_d_up_pmax_watt"] / 1e6
            ler_df["Cap_FcrdDo"] = ler_m_data["fcr_d_down_pmax_watt"] / 1e6
            ler_df.drop(columns=["timestamp"], inplace=True)
            ler_fn = get_output_fn(df, svk_df, args, test_type="LERTestLLHD")

            print(f"Writing to {ler_fn}.csv")
            ler_df.to_csv(ler_fn + ".csv", index=False, lineterminator='\r\n')

    print(f"Writing to {fn}.csv")
    svk_df.to_csv(fn + ".csv", index=False, lineterminator='\r\n')
