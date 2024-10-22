#!/usr/bin/env python3
# Generate plots proving stability of the FCR system given sine test results

import argparse
import logging
import math
from typing import List

from matplotlib import patches
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

j = 0+1j
T_FME = 1.0  # Frequency measurement delay [s]


def F_FME(s):
    """Frequency measurement equipment transfer function, FCR Technical Requirements section 4.4
    """
    # Exponential decay with 1 second time constant
    return 1 / (T_FME * s + 1)
    # return math.e ** (-T_FME*s)  # compare to a simple delay (phase shift)


# Transfer function of the power system, FCR Technical Requirements section 3.2 (equation 11)
def G(s):
    """Power system model, parameters from table 7 (equation 11)
    """
    DP_FCR = 1450  # FCR-D volume [MW]
    Df_FCR = 0.4  # FCR-D one-sided frequency band [Hz]
    f_0 = 50.0  # Nominal frequency [Hz]
    S_n = 23000  # Nominal power [MW]
    H = 120000 / S_n  # Inertia constant [s]
    K_f = 0.01  # Load frequency dependence
    return (DP_FCR/Df_FCR)*(f_0/S_n)*(1/(2*H*s + K_f*f_0))


def G_FCRN(s):
    """Power system model, parameters from table 7 (equation 11)
    """
    DP_FCR = 600  # FCR-N volume [MW]
    Df_FCR = 0.1  # FCR-N one-sided frequency band [Hz]
    f_0 = 50.0  # Nominal frequency [Hz]
    S_n = 42000  # Nominal power [MW]
    H = 190000 / S_n  # Inertia constant [s]
    K_f = 0.01  # Load frequency dependence
    return (DP_FCR/Df_FCR)*(f_0/S_n)*(1/(2*H*s + K_f*f_0))


def nyquist_plot(test_name, test_periods, G_O):
    # Angular frequencies at which to plot [rad/s]
    plot_w = [2*math.pi/p for p in test_periods]

    # Nyquist plot of the open-loop system (green curve)
    # Requirement 8: Stay outside the blue circle
    nyquist_points = [G_O(j*w) for w in plot_w]
    fig, ax = plt.subplots()
    ax.plot([n.real for n in nyquist_points], [n.imag for n in nyquist_points], '.-', color='green')
    ax.plot([0, nyquist_points[0].real], [0, nyquist_points[0].imag], '.', color='green', linestyle='dashed')
    ax.axis('equal')
    ax.add_patch(patches.Circle((-1, 0), radius=0.43, facecolor="none", edgecolor="b"))
    ax.set(title=f'Stability Nyquist ({test_name})')
    ax.grid()


def frequency_domain_performance_plot(test_name, test_periods, G_C):
    # Requirement 9: stay below the dashed line
    # FCR Technical Requirements, section 3.3
    def D(s):
        """Typical disturbance profile of the system"""
        return 1/(70*s+1)

    plot_w = [2 * math.pi / p for p in test_periods]
    fig, ax = plt.subplots()
    ax.loglog(test_periods, [abs(1 / D(j * w)) for w in plot_w], color='orange', linestyle='dashed')
    ax.loglog(test_periods, [abs(G_C(j * w)) for w in plot_w], color='orange')
    ax.set(xlabel='Time period [s]', ylabel='Magnitude', title=f'Frequency domain performance ({test_name})')
    ax.grid()


def crunch(test_power: float,
           test_name: str,
           test_periods: List[float],
           test_phase_shift_deg: List[float],
           test_amplitude: List[float],
           nyquist_file: str,
           frequency_performance_file: str):
    def F(s):
        """Normalized transfer function of system under test (equation 9)
        """
        # Find the index of the period given by s=j𝜔. 𝜔=2𝜋/T → T=2𝜋/𝜔
        i = test_periods.index(round(2 * math.pi / s.imag))
        phi = test_phase_shift_deg[i] / 180 * math.pi
        A = (test_amplitude[i] / 0.1) * (0.4 / test_power)
        return F_FME(s) * (A * math.cos(phi) + A * j * math.sin(phi))

    def G_O(s):
        """Open loop system (equation 10)"""
        return F(s) * G(s)

    def G_C(s):
        """Closed loop system"""
        K_margin = 0.95
        return K_margin * (G(s) / (1 + F(s) * G(s)))

    nyquist_plot(test_name, test_periods, G_O)
    if nyquist_file:
        plt.savefig(nyquist_file)

    frequency_domain_performance_plot(test_name, test_periods, G_C)
    if frequency_performance_file:
        plt.savefig(frequency_performance_file)


def run():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='FCR Stability plots')
    parser.add_argument('-t', nargs='+', type=float, help='Sine period [s]')
    parser.add_argument('-p', nargs='+', type=float, help='Phase shift [°]')
    parser.add_argument('-a', nargs='+', type=float, help='Amplitude [MW]')
    parser.add_argument('-f', type=float, help='Test FCR-D power [MW]', required=True)
    parser.add_argument('-n', help='Test name', required=True)
    parser.add_argument('--t-fme', type=float, help='T_FME Frequency measurement delay [1.0]', default=1.0)
    parser.add_argument('--nyquist', help='Nyquist plot file')
    parser.add_argument('--performance', help='Performance plot file')
    args = parser.parse_args()

    global T_FME
    T_FME = args.t_fme
    crunch(args.f, args.n, args.t, args.p, args.a, args.nyquist, args.performance)


if __name__ == '__main__':
    run()
