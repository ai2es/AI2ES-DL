"""
Schedules are functions that take integer step numbers and convert them to real values OR
functions take take values on the interval from [0-1] and convert them to real values
"""

import tensorflow as tf
from math import sin, pi


class BleedOut:
    # Oscillating learning rate schedule
    # inspired by https://arxiv.org/abs/1506.01186
    def __init__(self, lrate, minimum=1e-3):
        self.lrate = lrate
        self.minimum = minimum

    def __call__(self, index):
        x = index + 1
        frac = (1 - self.minimum) / x ** (1 - sin(2 * pi * (x ** .5)))
        return min(self.lrate * (frac + self.minimum), 1)


class Cyclic25:
    """CAI Cyclical and Advanced Learning Rate Scheduler.
    # Arguments
     	index: integer with current epoch count.
    # Returns
        float with desired learning rate.
    """

    def __init__(self, lrate):
        self.lrate = lrate

    def __call__(self, index):
        epoch = index + 1
        base_learning = self.lrate
        local_epoch = epoch % 25
        if local_epoch < 7:
            return base_learning * (1 + 0.5 * local_epoch)
        else:
            return (base_learning * 4) * (0.85 ** (local_epoch - 7))


def diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    """
    Linear diffusion schedule

    :param diffusion_times: number of diffusion steps possible
    :param min_signal_rate: Min signal rate (1 - max noise rate)
    :param max_signal_rate: Max signal rate (1 - min noise rate)
    """
    # diffusion times -> angles
    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    # angles -> signal and noise rates
    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)
    # note that their squared sum is always: sin^2(x) + cos^2(x) = 1
    # this is theoretically a necessary property, although I suspect only if you are using MSE

    return noise_rates, signal_rates
