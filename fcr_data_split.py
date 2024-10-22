#!/usr/bin/env python3
# Split up FCR-D test data into separate files, based on metadata

import argparse
import csv
import dataclasses
import json
import datetime
import logging
from contextlib import suppress
from typing import List, Dict, Tuple
import fcr_sine_file_cruncher
import fcr_ramp_file_cruncher
import fcr_sine_stability
import svk_ler_plot
from tabulate import tabulate
import colorama

log = logging.getLogger('splitter')
END_TRIM = 15  # Data to cut off at the end of test to avoid overlapping with next [s]
START_TRIM = 2  # Data to cut off at the start to avoid overlapping with previous test [s]
RAMP_START_TRIM = -2  # Include a bit of the 50 Hz before the test to please FCP IT Tool [s]


@dataclasses.dataclass
class FcrTestParams:
    power: int
    fcr_d_up: int
    fcr_d_down: int
    fcr_n: int
    ffr: int


@dataclasses.dataclass
class MetadataEntry:
    fcr_test_start: float
    fcr_test_params: FcrTestParams
    fcr_test_sequence: str
    fcr_test_sine_period: float
    fcr_test_end: float = None

    def __init__(self, fcr_test_start, fcr_test_params, fcr_test_sequence, fcr_test_sine_period=0.0, **kwargs):
        self.fcr_test_start = float(fcr_test_start)
        self.fcr_test_params = FcrTestParams(*json.loads(fcr_test_params))
        self.fcr_test_sequence = fcr_test_sequence
        self.fcr_test_sine_period = float(fcr_test_sine_period or 0.0)

    def __repr__(self):
        return f'{self.fcr_test_start}-{self.fcr_test_end} {self.fcr_test_sequence} '\
               f'{self.fcr_test_sine_period} {self.fcr_test_params}'


def clean_file_data(file_data: List[Dict]) -> List[Dict]:
    """Remove lines with undefined values, resulting from join operations in Grafana"""
    return list(filter(lambda d: 'undefined' not in d.values(), file_data))


def find_test_stages(meta_data) -> List[MetadataEntry]:
    entries = []
    for line in meta_data:
        with suppress(ValueError):
            entry = MetadataEntry(**line)
            if not len(entries) or entry != entries[-1]:
                entries.append(entry)
    return entries


def ts(s: str) -> float:
    """String to unix timestamp in UTC"""
    try:
        return datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc).timestamp()
    except ValueError:
        log.error(f'Value error when parsing datetime {s}')
        log.error('This usually means that you aren\'t running a new enough Python, 3.11 is needed')
        raise


def split_into_chunks(file_data: List[Dict], meta_data: List[Dict]) -> List[Tuple[MetadataEntry, List[Dict]]]:
    stages = find_test_stages(meta_data)
    for i, stage in enumerate(stages[:-1]):
        stage.fcr_test_end = stages[i+1].fcr_test_start
    file_data_end = ts(file_data[-1]['DateTime'])
    stages[-1].fcr_test_end = file_data_end

    log.debug(f'Found unique stages {stages}')

    chunks = []
    for stage in stages:
        log.debug(f'Stage {stage}')
        if 'sine' in stage.fcr_test_sequence and not stage.fcr_test_sine_period:
            # Skip zero-period sine, that's an artifact of the test sequence
            continue

        if 'ramp' in stage.fcr_test_sequence:
            start_trim = RAMP_START_TRIM
        else:
            start_trim = START_TRIM
        end_trim = END_TRIM

        stage_data = [row for row in file_data if
                      (stage.fcr_test_start + start_trim) <= ts(row['DateTime']) < (stage.fcr_test_end - end_trim)]
        chunks.append((stage, stage_data))

    return chunks


def export_chunk(stage: MetadataEntry, file_data: List[Dict], system_name: str, area: str) -> str | None:
    if len(file_data) < 100:
        log.warning(f'No data for {stage}, skipping')
        return None

    match stage.fcr_test_sequence:
        case 'fcr-n-step':
            service = 'Fcrn'
            test_type = 'RampTest'
            load_and_droop = 'HLHD'
        case 'fcr-d-up-ramp' | 'fcr-d-up-ramp-endurance':
            service = 'FcrdUp'
            test_type = 'RampTest'
            load_and_droop = 'HLHD'
        case 'fcr-d-down-ramp' | 'fcr-d-down-ramp-endurance':
            service = 'FcrdDo'
            test_type = 'RampTest'
            load_and_droop = 'HLHD'
        case 'fcr-n-sine':
            service = 'Fcrn'
            test_type = f'SineResponse{stage.fcr_test_sine_period:.0f}s'
            load_and_droop = 'HLHD'
        case 'fcr-d-up-sine':
            service = 'FcrdUp'
            test_type = f'SineResponse{stage.fcr_test_sine_period:.0f}s'
            load_and_droop = 'HLLD'
        case 'fcr-d-down-sine':
            service = 'FcrdDo'
            test_type = f'SineResponse{stage.fcr_test_sine_period:.0f}s'
            load_and_droop = 'HLLD'
        case 'ler-fcr-d-up' | 'fcr-d-up-ler':
            service = 'FcrdUp'
            test_type = 'LERTest'
            load_and_droop = 'LLHD'
        case 'ler-fcr-d-down' | 'fcr-d-down-ler':
            service = 'FcrdDo'
            test_type = 'LERTest'
            load_and_droop = 'LLHD'
        case _:
            log.error(f'Unknown test {stage}')
            return None

    timeformat_filename = '%Y%m%dT%H%M'
    start_date = datetime.datetime.fromisoformat(file_data[0]['DateTime']).strftime(timeformat_filename)
    end_date = datetime.datetime.fromisoformat(file_data[-1]['DateTime']).strftime(timeformat_filename)
    file_name = f'{system_name}_{service}_{test_type}{load_and_droop}_{area}_UTC_{start_date}-{end_date}_100ms.csv'

    log.info(f'Writing {file_name}, {len(file_data)} rows')
    with open(file_name, 'w', encoding='ascii') as f:
        writer = csv.DictWriter(f, fieldnames=file_data[0].keys())
        writer.writeheader()
        writer.writerows(file_data)

    return file_name


def get_sine_results_table(results: List[Tuple[MetadataEntry, fcr_sine_file_cruncher.Result]]) -> str:
    return tabulate([
        [result.period,
         result.stationary_periods,
         result.amplitude,
         result.normalized_amplitude,
         result.phase_shift,
         result.rmse,
         ]
        for stage, result in results],
        headers=['Period [s]',
                 'Stationary periods',
                 'Amplitude A_P [MW]',
                 'Amplitude norm.',
                 'Phase shift φ [°]',
                 'Linearity RMSE [%]'],
        floatfmt=['.04f', '.0f', '.04f', '.04f', '.02f', '.03f'])


def get_ramp_results_table(stage: MetadataEntry, result: fcr_ramp_file_cruncher.Result) -> str:
    test_power = 0
    match stage.fcr_test_sequence:
        case 'fcr-d-up-ramp' | 'fcr-d-up-ramp-endurance':
            test_power = stage.fcr_test_params.fcr_d_up / 1e6
        case 'fcr-d-down-ramp' | 'fcr-d-down-ramp-endurance':
            test_power = -stage.fcr_test_params.fcr_d_down / 1e6

    ramp_response = tabulate(
        [[test_power,
          0,
          result.P_ss3,
          result.P_ss4,
          result.delta_P_7_5s,
          result.energy_7_5s,
          ]],
        headers=['Theoretical P_ss,3 [MW]',
                 'Theoretical P_ss,4 [MW]',
                 'P_ss,3 [MW]',
                 'P_ss,4 [MW]',
                 '𝛥𝑃_7.5𝑠 [MW]',
                 'E_7.5s [MJ]'],
        floatfmt=['.04f', '.0f', '.04f', '.04f', '.04f', '.04f'])

    red_factor = 1
    if abs((result.P_ss3 - test_power) / test_power) > 5e-2:
        red_factor = 'not 1'
    steady_state_error = tabulate(
        [[result.P_ss3 - test_power,
          red_factor,
          result.P_ss3 - test_power]],
        headers=['Steady state error [MW]',
                 'Reduction factor',
                 'Steady state error with reduction factor[MW]'],
        floatfmt='.04f')

    red_factor = 1
    if (abs(result.delta_P_7_5s) / 0.86) < abs(test_power) or (abs(result.energy_7_5s) / 3.2) < abs(test_power):
        red_factor = 'not 1'
    energy_check = tabulate(
        [[
            abs(result.delta_P_7_5s) / 0.86,
            abs(result.energy_7_5s) / 3.2,
            abs(test_power),
            red_factor
          ]],
        headers=['𝛥𝑃_7.5𝑠 / 0.86',
                 'E_7.5s / 3.2',
                 '𝛥𝑃_ss,theoretical [MW]',
                 'K_red,dyn'],
        floatfmt='.03f')

    return '\n\n'.join((ramp_response, steady_state_error, energy_check))


def styled(s, style=None):
    if style is None:
        return s
    return style + s + colorama.Style.RESET_ALL


class ColorLogFormatter(logging.Formatter):
    styles = {
        logging.WARNING: colorama.Fore.YELLOW,
        logging.ERROR: colorama.Fore.RED,
        logging.FATAL: colorama.Fore.RED,
    }

    def format(self, record):
        style = self.styles.get(record.levelno, None)
        return styled(super().format(record), style)


def run():
    colorama.init()
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(ColorLogFormatter('%(levelname)s %(name)s %(message)s'))
    logging.basicConfig(level=logging.INFO, handlers=[log_handler])
    parser = argparse.ArgumentParser(description='FCR data file splitter')
    parser.add_argument('data_file', help='FCR-D test data CSV file')
    parser.add_argument('metadata_file', help='Test metadata CSV file')
    parser.add_argument('-n', '--system-name', help='System name (for export file name)', required=True)
    parser.add_argument('-a', '--area', help='Elområde [SE3]', default='SE3')
    parser.add_argument('--no-crunch', action='store_true', help='Do not interpret data')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.debug:
        log.setLevel(logging.DEBUG)

    exported_files = []

    # Read CSV files. Decode as utf-8-sig to strip the BOM added by Grafana
    with (open(args.data_file, encoding='utf-8-sig') as df,
          open(args.metadata_file, encoding='utf-8-sig') as mf):
        file_data = list(csv.DictReader(df))
        file_data = clean_file_data(file_data)
        meta_data = list(csv.DictReader(mf))
        chunks = split_into_chunks(file_data, meta_data)
        for stage, data in chunks:
            file_name = export_chunk(stage, data, args.system_name, args.area)
            if file_name:
                exported_files.append((stage, file_name))

    fcr_d_up_sine_results = []
    fcr_d_down_sine_results = []
    ramp_results = []
    if not args.no_crunch:
        for stage, file_name in exported_files:
            try:
                if 'sine' in stage.fcr_test_sequence:
                    match stage.fcr_test_sequence:
                        case 'fcr-d-up-sine':
                            test_power = stage.fcr_test_params.fcr_d_up / 1e6
                            result = fcr_sine_file_cruncher.do_file(file_name, test_power)
                            fcr_d_up_sine_results.append((stage, result))
                        case 'fcr-d-down-sine':
                            test_power = stage.fcr_test_params.fcr_d_down / 1e6
                            result = fcr_sine_file_cruncher.do_file(file_name, test_power)
                            fcr_d_down_sine_results.append((stage, result))
                        case _:
                            log.warning(f'Not handling sine test {stage.fcr_test_sequence}')

                if 'ramp' in stage.fcr_test_sequence or 'step' in stage.fcr_test_sequence:
                    result = fcr_ramp_file_cruncher.do_file(file_name)
                    ramp_results.append((stage, result))

                if 'ler' in stage.fcr_test_sequence:
                    plot_file = f'{stage.fcr_test_sequence} {args.system_name}.png'
                    log.info(f'Writing {plot_file}')
                    df = svk_ler_plot.parse_file(open(file_name, 'r'))
                    svk_ler_plot.plot(df, plot_file, args.system_name)

            except Exception:
                log.exception(f'Exception interpreting stage {file_name}')

    for sine_data in [fcr_d_up_sine_results, fcr_d_down_sine_results]:
        if sine_data:
            test_power = sine_data[0][1].test_power
            test_sequence = sine_data[0][0].fcr_test_sequence
            test_periods = [result.period for stage, result in sine_data]
            test_phase_shift_deg = [result.phase_shift for stage, result in sine_data]
            test_amplitude = [result.amplitude for stage, result in sine_data]
            nyquist_file = f'{test_sequence} {args.system_name} {test_power} MW nyquist.png'
            frequency_performance_file = f'{test_sequence} {args.system_name} {test_power} MW performance.png'

            log.info(f'Writing {nyquist_file}')
            log.info(f'Writing {frequency_performance_file}')

            fcr_sine_stability.crunch(test_power,
                                      f'{args.system_name} {test_power} MW',
                                      test_periods,
                                      test_phase_shift_deg,
                                      test_amplitude,
                                      nyquist_file,
                                      frequency_performance_file)

    print()
    print('━' * 120)
    print(styled(f'FCR-D test report for {args.system_name}', colorama.Style.BRIGHT))

    if fcr_d_up_sine_results:
        print()
        print(styled('FCR-D-up sine results', colorama.Back.BLUE))
        print(get_sine_results_table(fcr_d_up_sine_results))

    if fcr_d_down_sine_results:
        print()
        print(styled('FCR-D-down sine results', colorama.Back.BLUE))
        print(get_sine_results_table(fcr_d_down_sine_results))

    for stage, result in ramp_results:
        print()
        print(styled(f'{stage.fcr_test_sequence} ramp results', colorama.Back.GREEN))
        print(get_ramp_results_table(stage, result))


if __name__ == '__main__':
    run()
