#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import argparse


def parse_file(f):
    print(f"Parsing {f.name}...")

    df = pd.read_csv(f, parse_dates=[0], date_format="%Y%m%dT%H%M%S.%f")
    df.sort_values("DateTime", inplace=True)

    return df


def plot(df, filename, system_name='', plot_aem_fraction=False):
    "Plot LER graph given nomimal power in MW and extended data rame"

    print("Plotting LER graph...")

    # If the grid frequency is below 50 Hz, the plot should be FCR-D Up, otherwise FCR-D Down.
    is_up = False
    if df["GridFreq"].min() < 50:
        is_up = True

    # Create helper columns and canonicalize the data
    if is_up:
        if "Cap_FcrdUp" in df:
            nom_p = df["Cap_FcrdUp"].iloc[0]
        else:
            nom_p = df["fcr_d_up_pmax_watt"].iloc[0] / 1e6
        if "fcr_d_up_f_ref_milli_hz" in df:
            df["f_ref_up"] = df["fcr_d_up_f_ref_milli_hz"] / 1000
    else:
        if "Cap_FcrdDo" in df:
            nom_p = df["Cap_FcrdDo"].iloc[0]
        else:
            nom_p = df["fcr_d_down_pmax_watt"].iloc[0] / 1e6
        if "fcr_d_down_f_ref_milli_hz" in df:
            df["f_ref_down"] = df["fcr_d_down_f_ref_milli_hz"] / 1000

    df["TimeRel"] = df["DateTime"] - df["DateTime"][0]
    df["NomPow"] = df["InsAcPow"] / nom_p
    if "SOC" in df:
        df["SOC"] = df["SOC"] / 100
    else:
        df["SOC"] = df["ResSize_FcrdUp"] / (df["ResSize_FcrdUp"] + df["ResSize_FcrdDo"])

    fig, ax1 = plt.subplots()
    ax1.grid(True)
    # Add xticks every 5 minutes, TimeRel is in nanoseconds.
    xticks = range(0, int(df["TimeRel"].max().value / (5 * 60 * 1e9) + 2))
    ax1.set_xlim(0, df["TimeRel"].max().value)
    ax1.set_xticks(
        list(map(lambda x: x * 5 * 60 * 1e9, xticks)), map(lambda x: x * 5, xticks)
    )
    ax1.set_xlabel("Time (minutes)")
    fig.set_figheight(6)
    fig.set_figwidth(max(xticks))
    ax1.set_title(f"{system_name} FCR-D Up P={nom_p} MW" if is_up else f"{system_name} FCR-D Down P={nom_p} MW")

    # Plot Grid Frequency and Frequency reference
    (l1,) = ax1.plot(df["TimeRel"], df["GridFreq"], color="C1", label="Frequency input")
    ax1.set_ylabel("Frequency (Hz)")
    l2 = None
    if is_up:
        if "f_ref_up" in df:
            (l2,) = ax1.plot(
                df["TimeRel"],
                df["f_ref_up"],
                color="C4",
                label="Frequency reference",
            )
        ax1.set_ylim((49.4, 50.1))
        ax1.set_yticks(np.arange(49.4, 50.1, 0.1))
    else:
        if "f_ref_down" in df:
            (l2,) = ax1.plot(
                df["TimeRel"],
                df["f_ref_down"],
                color="C4",
                label="Frequency reference",
            )
        ax1.set_ylim((49.9, 50.6))
        ax1.set_yticks(np.arange(49.9, 50.6, 0.1))

    # Plot AEM/NEM on/off markers
    pos = (-24, -17)
    if is_up:
        pos = (-24, 7)

    aem_on = df[df["AEM"].diff() == 1]
    for _, row in aem_on.iterrows():
        ax1.annotate(
            "AEM On",
            (row["TimeRel"] / pd.Timedelta(1, "ns"), row["GridFreq"]),
            xytext=pos,
            textcoords="offset points",
        )
        ax1.plot(row["TimeRel"], row["GridFreq"], "C1o")

    aem_off = df[df["AEM"].diff() == -1]
    for _, row in aem_off.iterrows():
        ax1.annotate(
            "AEM Off",
            (row["TimeRel"] / pd.Timedelta(1, "ns"), row["GridFreq"]),
            xytext=pos,
            textcoords="offset points",
        )
        ax1.plot(row["TimeRel"], row["GridFreq"], "C1o")

    nem_on = df[df["NEM"].astype(bool).astype(int).diff() == 1]
    for _, row in nem_on.iterrows():
        ax1.annotate(
            "NEM On",
            (row["TimeRel"] / pd.Timedelta(1, "ns"), row["GridFreq"]),
            xytext=pos,
            textcoords="offset points",
        )
        ax1.plot(row["TimeRel"], row["GridFreq"], "C1o")

    nem_off = df[df["NEM"].astype(bool).astype(int).diff() == -1]
    for _, row in nem_off.iterrows():
        ax1.annotate(
            "NEM Off",
            (row["TimeRel"] / pd.Timedelta(1, "ns"), row["GridFreq"]),
            xytext=pos,
            textcoords="offset points",
        )
        ax1.plot(row["TimeRel"], row["GridFreq"], "C1o")

    # Plot Nominal power output and SOC
    ax2 = ax1.twinx()
    (l3,) = ax2.plot(
        df["TimeRel"], df["NomPow"], color="C0", label="Active power output"
    )
    (l4,) = ax2.plot(df["TimeRel"], df["SOC"], color="C7", label="State of charge")
    ax2.set_ylabel("Active power output / SOC (%)")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    ax2.set_ylim((-1.05, 1.05))
    ax2.set_yticks([n / 100 for n in range(-100, 101, 10)])

    if plot_aem_fraction:
        (l5,) = ax2.plot(df["TimeRel"], df["AEM"], color="C8", label="Fraction in AEM")
    else:
        l5 = None

    ax1.legend(handles=list(filter(None, [l1, l2, l3, l4, l5])), ncol=4, frameon=False, loc="lower right")
    fig.savefig(filename, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        type=argparse.FileType("r"),
        help="File from datalog server, - for stdin.",
    )
    parser.add_argument(
        "-o", "--output", help="Output file", type=str, default="ler_plot.png")
    parser.add_argument(
        "--aem-fraction", action="store_true", help="Plot fractional AEM status"
    )
    parser.add_argument(
        "-n", help="System name"
    )

    args = parser.parse_args()

    df = parse_file(args.filename)
    plot(df, args.output, system_name=args.n or "", plot_aem_fraction=args.aem_fraction)
