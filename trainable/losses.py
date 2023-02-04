"""Custom loss functions for neural network training"""

import tensorflow as tf


def CE(y_true, y_pred):
    """Cross-entropy for probability vector outputs"""
    return tf.math.negative(tf.reduce_sum(tf.math.log(y_pred) * y_true, axis=-1))


def NCE(y_true, y_pred):
    """
    'negative' cross-entropy for probability vector outputs. Simply penalizes confidence in a prediction

    ln(mean(y_pred) / max(y_pred))
    """
    return tf.math.negative(
        tf.math.log(tf.reduce_mean(y_pred, axis=-1)) - tf.math.log(tf.reduce_max(y_pred, axis=-1)))


def logsum(y_true, y_pred):
    """
    penalizes model for predicting any probabilities in the y_pred vector so long as they do not always sum
    to a constant value (such as 1).
    ln([1 - sum(y_pred)] + \\epsilon)
    """
    return tf.math.negative(tf.math.log(1 + 2 ** (-16) - tf.reduce_sum(y_pred, keepdims=True, axis=-1)))


def identity(y_true, y_pred):
    """
    loss function that is just the identity transformation
    """
    return y_pred
