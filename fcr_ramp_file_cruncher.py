#!/usr/bin/env python3
# Find features in FCR-D ramp test data and calculate some stats

import argparse
import csv
import dataclasses
from datetime import datetime, timedelta
import logging
import re
from pathlib import Path
from typing import List

import numpy
import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


@dataclasses.dataclass
class Frame:
    datetime: datetime
    active_power: float
    grid_freq: float

    def __init__(self, DateTime, InsAcPow, GridFreq, **kwargs):
        self.datetime = datetime.fromisoformat(DateTime)
        self.active_power = float(InsAcPow)
        self.grid_freq = float(GridFreq)


@dataclasses.dataclass
class Result:
    P_ss3: float  # Steady state power after ramp 3 [MW]
    P_ss4: float  # Steady state power after ramp 4 [MW]
    delta_P_7_5s: float  # Power 7.5s after spike [MW]
    energy_7_5s: float  # Energy 7.5s after spike [MWs]


def find_first(frames: List[Frame], condition):
    for f in frames:
        if condition(f):
            return f
    raise IndexError("Not found")


def crunch(frames: List[Frame], service, do_plot=False) -> Result:
    log.debug(f"Ramp with {len(frames)} data points")
    log.debug(f"Start time {frames[0].datetime}")
    log.debug(f"End time {frames[-1].datetime}")

    # Convert to numpy arrays for easier handling
    time_array = np.array([f.datetime for f in frames])
    freq_array = np.array([f.grid_freq for f in frames])
    power_array = np.array([f.active_power for f in frames])

    log.debug(f"Freq span: {min(freq_array)} - {max(freq_array)}")
    log.debug(f"Power span: {min(power_array)} - {max(power_array)}")

    # Always work in frequencies dipping below 50 Hz. Flip FCR-D-down to make the math
    # the same as for FCR-D-up
    def norm_f(f):
        if service == "FcrdDo":
            return 100 - f
        else:
            return f

    # Find ramps
    s = timedelta(seconds=1)
    ramp_1_start = find_first(frames, lambda frame: norm_f(frame.grid_freq) < 49.9)
    ramp_1_end = find_first(
        frames,
        lambda frame: (
            (ramp_1_start.datetime + 2 * s)
            < frame.datetime
            < (ramp_1_start.datetime + 7 * s)
            and norm_f(abs(frame.grid_freq - 49.45) <= 0.05)
        ),
    )
    ramp_2_end = find_first(
        frames,
        lambda frame: (
            ramp_1_end.datetime < frame.datetime < (ramp_1_end.datetime + 7 * s)
            and norm_f(abs(frame.grid_freq - 49.9) <= 0.05)
        ),
    )
    ramp_3_end = find_first(
        frames,
        lambda frame: (
            (ramp_2_end.datetime + 40 * s)
            < frame.datetime
            < (ramp_2_end.datetime + 60 * s)
            and norm_f(abs(frame.grid_freq - 49.5) <= 0.05)
        ),
    )
    ramp_4_start = find_first(
        frames,
        lambda frame: (
            (ramp_3_end.datetime + 290 * s)
            < frame.datetime
            < (ramp_3_end.datetime + 910 * s)
            and norm_f(frame.grid_freq) > 49.5
        ),
    )
    ramp_4_end = find_first(
        frames,
        lambda frame: (
            (ramp_4_start.datetime + 1 * s)
            < frame.datetime
            < (ramp_4_start.datetime + 4 * s)
            and norm_f(abs(frame.grid_freq - 49.9) <= 0.05)
        ),
    )
    ramp_5_start = find_first(
        frames,
        lambda frame: (
            (ramp_4_end.datetime + 290 * s)
            < frame.datetime
            < (ramp_4_end.datetime + 310 * s)
            and norm_f(frame.grid_freq) < 49.9
        ),
    )
    ramp_5_end = find_first(
        frames,
        lambda frame: (
            (ramp_5_start.datetime + 3 * s)
            < frame.datetime
            < (ramp_5_start.datetime + 6 * s)
            and norm_f(abs(frame.grid_freq - 49.0) <= 0.05)
        ),
    )
    print(ramp_4_end)

    # Energy after ramp 1 (requirement 4)
    zenith_time = ramp_1_start.datetime + 4.4 * s
    frame_at_zenith = find_first(
        frames, lambda frame: (zenith_time <= frame.datetime < zenith_time + 0.2 * s)
    )
    log.info(f"Power at spike zenith/nadir ∆P_zenith (or ∆P_nadir) {frame_at_zenith}")
    # Find the worst energy from 0 to 40 seconds after zenith (requirement 4). Assuming power > 0.5*|∆P_ss,theo|
    worst_energy = 0
    for i in range(0, 40):
        if service == "FcrdUp":
            sign = 1
        elif service == "FcrdDo":
            sign = -1
        integration_frames = [
            frame
            for frame in frames
            if zenith_time <= frame.datetime < zenith_time + i * s
        ]
        energy = numpy.trapz(
            y=[
                sign * frame.active_power - abs(frame_at_zenith.active_power)
                for frame in integration_frames
            ],
            x=[frame.datetime.timestamp() for frame in integration_frames],
        )
        worst_energy = max(energy, worst_energy)
    log.info(f"Max excess energy after spike (req 4) {worst_energy:.4f} MJ")

    # Calculate steady state power after ramp 3 and 4
    P_ss3 = np.average(
        [
            frame.active_power
            for frame in frames
            if (ramp_3_end.datetime + 10 * s)
            < frame.datetime
            < (ramp_4_start.datetime - 10 * s)
        ]
    )
    P_ss4 = np.average(
        [
            frame.active_power
            for frame in frames
            if (ramp_4_end.datetime + 10 * s)
            < frame.datetime
            < (ramp_5_start.datetime - 10 * s)
        ]
    )
    lowest_maintained_power = np.min(
        [
            abs(frame.active_power)
            for frame in frames
            if (ramp_3_end.datetime + 2 * s)
            < frame.datetime
            < (ramp_4_start.datetime - 2 * s)
        ]
    )

    log.info(f"Steady state active power P_ss,3={P_ss3:.3f} MW")
    log.info(f"Steady state inactive power P_ss,4={P_ss4:.3f} MW")
    log.info(f"Steady state response {P_ss3-P_ss4:.5f} MW")

    if lowest_maintained_power < 0.98 * abs(P_ss3):
        log.warning(
            f"Failed to maintain power after ramp 3: {lowest_maintained_power:.3f} MW"
        )

    # Calculate power and energy after ramp 5, during 7.5s
    activation_frames = [
        frame
        for frame in frames
        if ramp_5_start.datetime <= frame.datetime < (ramp_5_start.datetime + 7.5 * s)
    ]
    assert norm_f(activation_frames[-1].grid_freq) == 49.0
    dP_7p5s = activation_frames[-1].active_power - P_ss4
    E_7p5s = numpy.trapz(
        y=[frame.active_power for frame in activation_frames],
        x=[frame.datetime.timestamp() for frame in activation_frames],
    )

    log.info(f"Activated power after 7.5s ∆P_7.5s={dP_7p5s:.3f} MW")
    log.info(f"Energy after 7.5s E_7.5s={E_7p5s:.3f} MJ")

    result = Result(P_ss3, P_ss4, dP_7p5s, E_7p5s)

    if do_plot:
        fig, axes = plt.subplots(nrows=2)
        axes[0].plot(time_array, freq_array, "y-", label="Frequency data")
        axes[0].annotate("1", xy=(ramp_1_start.datetime, ramp_1_start.grid_freq))
        axes[0].annotate("2", xy=(ramp_2_end.datetime, ramp_2_end.grid_freq))
        axes[0].annotate("3", xy=(ramp_3_end.datetime, ramp_3_end.grid_freq))
        axes[0].annotate("4s", xy=(ramp_4_start.datetime, ramp_4_start.grid_freq))
        axes[0].annotate("4e", xy=(ramp_4_end.datetime, ramp_4_end.grid_freq))
        axes[0].annotate("5s", xy=(ramp_5_start.datetime, ramp_5_start.grid_freq))
        axes[0].annotate("5e", xy=(ramp_5_end.datetime, ramp_5_end.grid_freq))
        axes[1].annotate(
            "z", xy=(frame_at_zenith.datetime, frame_at_zenith.active_power)
        )
        axes[1].plot(time_array, power_array, "b-", label="Power data")
        plt.show()

    return result


def do_file(filename, show_plot=False, debug_log=False) -> Result:
    match = re.match(r"(.*)_(.*)_RampTest...._.*_.*_.*_.*", Path(filename).stem)
    site, service = match.groups()

    global log
    log = logging.getLogger(f"{site} {service}")
    if debug_log:
        log.setLevel(logging.DEBUG)

    with open(filename) as f:
        file_data = csv.DictReader(f)
        frames = [Frame(**line) for line in file_data]
        return crunch(frames, service, show_plot)


def run():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="FCR Ramp file cruncher")
    parser.add_argument("file", nargs="+", help="FCR-D test ramp response CVS file[s]")
    parser.add_argument("--plot", action="store_true", help="Show plot")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    for filename in args.file:
        do_file(filename, args.plot, args.debug)


if __name__ == "__main__":
    run()
