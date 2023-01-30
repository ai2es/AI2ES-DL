import tensorflow as tf

def bleed_out(lrate=1e-3):
    def schedule(index, lrate=1e-3, minimum=1e-3):
        # Oscillating learning rate schedule
        # inspired by https://arxiv.org/abs/1506.01186
        from math import sin, pi
        x = index + 1
        frac = (1 - minimum) / x ** (1 - sin(2 * pi * (x ** .5)))
        return min(lrate * (frac + minimum), 1)
    return schedule


def cyclical_adv_lrscheduler25(lrate=1e-3):
    def schedule(epoch):
        """CAI Cyclical and Advanced Learning Rate Scheduler.
        # Arguments
            epoch: integer with current epoch count.
        # Returns
            float with desired learning rate.
        """
        base_learning = lrate
        local_epoch = epoch % 25
        if local_epoch < 7:
            return base_learning * (1 + 0.5 * local_epoch)
        else:
            return (base_learning * 4) * (0.85 ** (local_epoch - 7))
    return schedule


def diffusion_schedule(diffusion_times, min_signal_rate = 0.02, max_signal_rate = 0.95):
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
