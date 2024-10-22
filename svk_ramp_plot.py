#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os.path


def parse_file(f):
    print(f"Parsing {f.name}...")

    df = pd.read_csv(f, parse_dates=[0], date_format="%Y%m%dT%H%M%S.%f")
    df.sort_values("DateTime", inplace=True)

    return df


def plot(df, filename):
    "Plot ramp graph from standard SVK data rame"

    print("Plotting ramp graph...")

    # If the grid frequency is below 50 Hz, the plot should be FCR-D Up, otherwise FCR-D Down.
    is_up = False
    if df["GridFreq"].min() < 50:
        is_up = True

    # Create helper columns
    df["TimeRel"] = df["DateTime"] - df["DateTime"][0]

    fig, ax1 = plt.subplots()
    ax1.grid(True)
    # Add xticks every 5 minutes, TimeRel is in nanoseconds.
    xticks = range(0, int(df["TimeRel"].max().value / (5 * 60 * 1e9) + 1))
    ax1.set_xlim(0, df["TimeRel"].max().value)
    ax1.set_xticks(
        list(map(lambda x: x * 5 * 60 * 1e9, xticks)), map(lambda x: x * 5, xticks)
    )
    ax1.set_xlabel("Time (minutes)")
    fig.set_figheight(10)
    fig.set_figwidth(max(xticks) * 5)
    ax1.set_title(f"FCR-D Up ramp" if is_up else f"FCR-D Down ramp")

    # Plot Grid Frequency and Frequency reference
    (l1,) = ax1.plot(df["TimeRel"], df["GridFreq"], color="C1", label="Frequency input")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_ylim((49, 51))
    ax1.set_yticks([49.0, 49.2, 49.4, 49.6, 49.8, 50.0, 50.2, 50.4, 50.6, 50.8, 51.0])

    # Plot Nominal power output and SOC
    ax2 = ax1.twinx()
    (l2,) = ax2.plot(
        df["TimeRel"], df["InsAcPow"], color="C0", label="Active power output"
    )
    ax2.set_ylabel("Active power output (MW)")
    # Adjust the y-axis limits to show the data better, similar to the requirements
    if is_up:
        ax2.set_ylim((df["InsAcPow"].max() * -1.1 * 4 / 3, df["InsAcPow"].max() * 1.1))
    else:
        ax2.set_ylim((df["InsAcPow"].min() * 1.1, df["InsAcPow"].min() * -1.1 * 4 / 3))

    ax1.legend(handles=[l1, l2], ncol=2, frameon=False, loc="lower right")
    fig.savefig(filename, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        type=argparse.FileType("r"),
        help="File from datalog server, - for stdin.",
    )
    args = parser.parse_args()

    df = parse_file(args.filename)
    out_fn = os.path.splitext(args.filename.name)[0] + ".png"
    plot(df, out_fn)
