#!/usr/bin/env python3
# Fit sine wave on FCR-D sine test data, calculate some stats

import argparse
import csv
import dataclasses
import datetime
import logging
import math
import re
from collections import namedtuple
from pathlib import Path
from typing import List, Tuple
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

fit_tuple = namedtuple('fit_tuple', ['dc_offset', 'amplitude', 'phase'])


@dataclasses.dataclass
class Frame:
    datetime: datetime.datetime
    active_power: float
    grid_freq: float

    def __init__(self, DateTime, InsAcPow, GridFreq, **kwargs):
        self.datetime = datetime.datetime.fromisoformat(DateTime)
        self.active_power = float(InsAcPow)
        self.grid_freq = float(GridFreq)


@dataclasses.dataclass
class Result:
    period: float
    test_power: float  # [MW]
    amplitude: float  # [MW]
    normalized_amplitude: float  # [MW]
    phase_shift: float  # [°]
    rmse: float  # RMS error [%]
    stationary_periods: float


def sine_function(t, angular_freq, dc_offset, amplitude, phase):
    return dc_offset + amplitude * np.sin(t * angular_freq + phase)


def fit_sine(x, y, angular_freq) -> Tuple[fit_tuple, float]:
    """Fit a sine to data. Return fit parameters and normalized RMS error"""

    def fix_frequency_function(t, dc_offset, amplitude, phase):
        # functools.partial doesn't work with curve_fit(), so do this manually instead
        return sine_function(t, angular_freq, dc_offset, amplitude, phase)

    # Curve_fit will wiggle the parameters to the function (a sine wave) to
    # find the parameters that make it fit best with the input data
    # (a measured sine wave). It is fairly fragile and I couldn't get it to
    # work if asked to fit the frequency. The way we use it we know the
    # exact frequency from the period of the test signal, do there is never
    # any doubt of the frequency to fit to.
    fit, _ = curve_fit(fix_frequency_function, x, y,
                       p0=(np.average(y), max(y)-np.average(y), 1.0),
                       bounds=((-math.inf, 0, -math.tau), (math.inf, math.inf, math.tau)),
                       method='trf', diff_step=1e-3)
    fit = fit_tuple(*fit)
    # If it found a negative phase, normalize it to the 0-2pi range
    if fit.phase < 0:
        fit = fit_tuple(fit.dc_offset, fit.amplitude, fit.phase + math.tau)
    log.debug(f'Fit: dc offset {fit.dc_offset:.3f}, amplitude {fit.amplitude:.3f}, phase {fit.phase:.3f} rad')

    # Calculate normalized RMS error, as specified in Requirement 10
    fit_data = fix_frequency_function(x, *fit)
    errors = y - fit_data
    fit_stddev = np.std(fit_data)
    rmse = math.sqrt(np.average(errors ** 2)) / fit_stddev

    return fit, rmse


def crunch(frames: List[Frame], period: float, fcr_power_setting: float, do_plot=False) -> Result:
    log.debug(f'Sine with {len(frames)} data points, {period} s period')
    log.debug(f'Start time {frames[0].datetime}')
    log.debug(f'End time {frames[-1].datetime}')

    # Convert to numpy arrays for easier handling
    time_array = np.array([f.datetime.timestamp() for f in frames])
    freq_array = np.array([f.grid_freq for f in frames])
    power_array = np.array([f.active_power for f in frames])

    log.debug(f'Freq span: {min(freq_array)} - {max(freq_array)}')
    log.debug(f'Power span: {min(power_array)} - {max(power_array)}')

    # Fit sines to the frequency and power data by wiggling amplitude and phase
    angular_freq = 2*math.pi/period
    freq_fit, freq_rmse = fit_sine(time_array, freq_array, angular_freq)
    power_fit, power_rmse = fit_sine(time_array, power_array, angular_freq)

    # Make sure that we actually found the data
    if freq_rmse > 0.05:
        log.warning(f'Bad frequency fit, RMSE {freq_rmse*100:.3}%')

    phase_diff = freq_fit[2] - power_fit[2] + (math.pi if power_fit[1] < 0 else 0)
    if phase_diff < 0:
        phase_diff += math.tau
    result = Result(period,
                    fcr_power_setting,
                    power_fit[1],
                    power_fit[1] / fcr_power_setting,
                    phase_diff / math.tau * 360,
                    power_rmse * 100,
                    (frames[-1].datetime - frames[0].datetime).seconds / period)

    log.info(f'Period T={result.period}, power amplitude A_P={result.amplitude:.3} MW, '
             f'phase shift φ={result.phase_shift:.1f}°, RMSE {result.rmse:.3f}%')

    if do_plot:
        fig, axes = plt.subplots(nrows=2)
        axes[0].plot(time_array, freq_array, 'b.', label='Frequency data')
        axes[0].plot(time_array, sine_function(time_array, angular_freq, *freq_fit), 'r-', label='Fit curve')
        axes[0].legend()
        axes[1].plot(time_array, power_array, 'b.', label='Power data')
        axes[1].plot(time_array, sine_function(time_array, angular_freq, *power_fit), 'r-', label='Fit curve')
        axes[1].legend()
        plt.show()

    return result


def do_file(filename, fcr_power_setting: float, show_plot=False):
    match = re.match(r'(.*)_(.*)_SineResponse(.*)s...._.*_.*_.*_.*', Path(filename).stem)
    site, service, _ = match.groups()
    period = int(match.group(3))

    global log
    log = logging.getLogger(f'{site} {service} {period}s')
    # log.setLevel(logging.DEBUG)

    with open(filename) as f:
        file_data = csv.DictReader(f)
        frames = [Frame(**line) for line in file_data]
        return crunch(frames, period, fcr_power_setting, show_plot)


def run():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='FCR Sine file cruncher')
    parser.add_argument('file', nargs='+', help='FCR-D test sine response CVS file[s]')
    parser.add_argument('-p', type=float, help='Theoretical FCR power [MW]', required=True)
    parser.add_argument('--plot', action='store_true', help='Show plot')
    args = parser.parse_args()

    for filename in args.file:
        do_file(filename, args.p, args.plot)


if __name__ == '__main__':
    run()
