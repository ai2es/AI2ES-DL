"""
Activation functions for neural network layers
"""

import tensorflow as tf


def hardswish(x):
    """Hardswish activation function"""
    return x * tf.nn.relu6(x + 3) / 6


def mish(x):
    """Mish activation function"""
    return x * tf.nn.tanh(tf.nn.softplus(x))


def int_act_slip(x):
    """
    activation function that just pushes every real number closer to its closest integer (except x % 1 = .5)
    this step function has slippery steps, e.g. the function is monotonic increasing.
    """

    s = .1
    m = tf.math.negative(tf.ones_like(x)) * (1 + s)
    m = tf.math.reciprocal(m)
    xmod1 = tf.math.floormod(x, tf.ones_like(x))
    # three distinguished cases:
    # if xmod1 <= (.5 - s/2)
    cond1 = tf.math.less_equal(xmod1, .5 - (s / 2))

    # if (.5 - s/2) <= xmod1 <= (.5 + s/2)
    cond2 = tf.math.logical_and(tf.math.greater(xmod1, .5 - s / 2), tf.math.less(xmod1, .5 + s / 2))

    # if (.5 + s/2) < xmod1
    cond3 = tf.math.greater_equal(xmod1, .5 + (s / 2))

    x = tf.where(cond1, x + (xmod1 * m), x)
    x = tf.where(cond2, x + ((xmod1 - .5) * ((1 / s) - 1) / (s + 1)), x)
    x = tf.where(cond3, x + (m * (xmod1 - 1)), x)

    return x


def int_act_no_slip(x):
    """
    activation function that just pushes every real number closer to its closest integer (except x % 1 = .5)
    this step function has slippery steps, e.g. the function is not monotonic increasing.
    """
    s = .1
    m = tf.math.negative(tf.ones_like(x) * (1 - s))
    m = tf.math.reciprocal(m)
    xmod1 = tf.math.floormod(x, tf.ones_like(x))
    # three distinguished cases:
    # if xmod1 <= (.5 - s/2)
    cond1 = tf.math.less_equal(xmod1, .5 - (s / 2))

    # if (.5 - s/2) <= xmod1 <= (.5 + s/2)
    cond2 = tf.math.logical_and(tf.math.greater(xmod1, .5 - s / 2), tf.math.less(xmod1, .5 + s / 2))

    # if (.5 + s/2) < xmod1
    cond3 = tf.math.greater_equal(xmod1, .5 + (s / 2))

    x = tf.where(cond1, x + (xmod1 * m), x)
    x = tf.where(cond2, x + ((xmod1 - .5) * (1 / s)), x)
    x = tf.where(cond3, x + (m * (xmod1 - 1)), x)

    return x
