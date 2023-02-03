""""
Mixed / Multi Sample Data Augmentation Methods.

data augmentations functions must require only an input dataset and return only a
tf.data.Dataset object representing the augmented dataset.  This module includes both
dataset returning and helper functions.
"""

import tensorflow as tf
import numpy as np
from optimization.data_augmentation.ssda import add_gaussian_noise, custom_rand_augment, get_spectrum, fftfreqnd


def blended_dset(train_ds, n_blended=2, prob=.5, std=.1, **kwargs):
    """
    Mixed sample data agumentation method that blends n_blended images via their arithmetic mean.  Weights
    are chosen from the uniform n_blended-probability simplex.  Gaussian noise with stdev std is added to each
    image before the mean is taken.

    :param train_ds: dataset of training images
    :param batch_size: size of batches to return from the generator
    :param n_blended: number of examples to blend together
    :param prob: probability of repacing a training batch with a convex combination of n_blended
    :param std: standard deviation of (mean 0) gaussian noise to add to images before blending
                (0.0 or equivalently None for no noise)

    :return: a tf.data.Dataset
    """
    # what if we take elements in our dataset and blend them together and predict the mean label?

    prob = prob if prob is not None else 1.0
    std = float(std) if std is not None else 0.0

    def add_gaussian_noise(x, y, std=1.0):
        return x + tf.random.normal(shape=x.shape, mean=0.0, stddev=std, dtype=tf.float32), y

    # create a dataset from which to get batches to blend
    dataset = train_ds.map(
        lambda x, y: tf.py_function(add_gaussian_noise, inp=[x, y, std], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE).batch(n_blended)

    def random_weighting(n):
        # get a random weighting chosen uniformly from the convex hull of the unit vectors.
        samp = -1 * np.log(np.random.uniform(0, 1, n))
        samp /= np.sum(samp)
        return np.array(samp)

    # the generator yields batches blended together with this weighting
    def blend(x, y):
        """
         sum a batch along the batch dimension weighted by a uniform random vector from the n simplex
          (convex hull of unit vectors)
        """
        if prob < np.random.uniform(0, 1, 1):
            return x[0], y[0]
        # compute the weights for the combination
        weights = random_weighting(n_blended)
        weights *= float(1 / np.linalg.norm(weights))
        weights = np.array(weights, dtype=np.double).reshape(-1, 1)
        # sum along the 0th dimension weighted by weights
        x = tf.tensordot(weights, tf.cast(x, tf.double), (0, 0))[0]
        y = tf.tensordot(weights, tf.cast(y, tf.double), (0, 0))[0]
        # return the convex combination
        return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    # map the dataset with the blend function
    dataset = dataset.map(lambda x, y: tf.py_function(blend, inp=[x, y], Tout=(tf.float32, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


def mixup_dset(train_ds, alpha=None, **kwargs):
    """
    mixup data augmentation strategy proposed in https://arxiv.org/abs/1710.09412.  Alpha is the parameter
    for a dirichlet distribution from which the weights of two examples are drawn before their weighted
    mean is returned.

    :param train_ds: dataset of batches to train on
    :param alpha: Dirichlet parameter.  Weights are drawn from Dirichlet(alpha, ..., alpha) for combining two examples.
                    Empirically choose a value in [.1, .4]

    :return: a tf.data.Dataset
    """
    # what if we take elements in our dataset and blend them together and predict the mean label?

    alpha = alpha if alpha is not None else 1.0

    rng = np.random.default_rng()

    # create a dataset from which to get batches to blend
    dataset = train_ds.batch(2)

    def random_weighting(n):
        return rng.dirichlet([alpha for i in range(n)], 1)

    # the generator yields batches blended together with this weighting
    # the generator yields batches blended together with this weighting
    def blend(x, y):
        """
         sum a batch along the batch dimension weighted by a uniform random vector from the n simplex
          (convex hull of unit vectors)
        """
        # compute the weights for the combination
        weights = random_weighting(2)
        weights *= float(1 / np.linalg.norm(weights))
        weights = np.array(weights, dtype=np.double).reshape(-1, 1)
        # sum along the 0th dimension weighted by weights
        x = tf.tensordot(weights, tf.cast(x, tf.double), (0, 0))[0]
        y = tf.tensordot(weights, tf.cast(y, tf.double), (0, 0))[0]
        # return the convex combination
        return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    # map the dataset with the blend function
    dataset = dataset.map(lambda x, y: tf.py_function(blend, inp=[x, y], Tout=(tf.float32, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


def bc_plus(train_ds, **kwargs):
    """
    The BC+ data augmentation technique improving upon the work published in https://www.frontiersin.org/articles/10.3389/fnhum.2021.765525/full.
    The weighted mean of two examples are taken with weights drawn from the uniform distribution.  Their weights are rescaled
    based on the standard deviation of the image tensors in the examples.  The examples are centered before they are combined.

    :param train_ds: dataset of batches to train on

    :return: a tf.data.Dataset
    """
    # what if we take elements in our dataset and blend them together and predict the mean label?

    rng = np.random.default_rng()

    # create a dataset from which to get batches to blend
    dataset = train_ds.batch(2)

    def random_weighting(sigma_1, sigma_2):
        p = rng.uniform(0, 1, 1)
        p = 1 / (1 + ((sigma_1 / sigma_2) * ((1 - p) / p)))
        return np.array([p, 1 - p])

    # the generator yields batches blended together with this weighting
    def blend(x, y):
        """
         sum a batch along the batch dimension weighted by a uniform random vector from the n simplex
          (convex hull of unit vectors)
        """
        # compute the weights for the combination
        weights = random_weighting(tf.sqrt(tf.math.reduce_variance(x[0])), tf.sqrt(tf.math.reduce_variance(x[1])))
        weights *= float(1 / np.linalg.norm(weights))
        weights = np.array(weights, dtype=np.double)
        # sum along the 0th dimension weighted by weights
        x = tf.tensordot(weights, tf.cast(x, tf.double), (0, 0))[0]
        y = tf.tensordot(weights, tf.cast(y, tf.double), (0, 0))[0]
        # return the convex combination
        return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    # map the dataset with the blend function
    dataset = dataset.map(lambda x, y: tf.py_function(blend, inp=[x, y], Tout=(tf.float32, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


def generalized_bc_plus(train_ds, n_blended=2, alpha=.25, std=0.05, **kwargs):
    """
    A version of the BC+ data augmentation technique improving upon the work published in
    https://www.frontiersin.org/articles/10.3389/fnhum.2021.765525/full.  Which has been generalized to more than two images.
    The weighted mean of two examples are taken with weights drawn from the dirichlet distribution parameterized by alpha.
    Their weights are not rescaled.  The examples are centered and noised before they are combined.


    :param train_ds: dataset of batches to train on
    :param n_blended: number of examples to mix
    :param alpha: Dirichlet parameter.  Weights are drawn from Dirichlet(alpha, ..., alpha) for combining two examples.
                    Empirically choose a value in [.1, .4]
    :param std: standard deviation of (mean 0) gaussian noise to add to images before blending
                (0.0 or equivalently None for no noise)
    :return: a dataset
    """
    # what if we take elements in our dataset and blend them together and predict the mean label?

    alpha = alpha if alpha is not None else 1.0

    rng = np.random.default_rng()

    def add_gaussian_noise(x, y, std=0.01):
        return x + tf.random.normal(shape=x.shape, mean=0.0, stddev=std, dtype=tf.float32), y

    # create a dataset from which to get batches to blend
    # add gaussian noise to each tensor
    dataset = train_ds.map(
        lambda x, y: tf.py_function(add_gaussian_noise, inp=[x, y, std], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE).batch(n_blended)

    def random_weighting(n):
        return rng.dirichlet(tuple([alpha for i in range(n)]), 1)

    # the generator yields batches blended together with this weighting
    def blend(x, y):
        """
         sum a batch along the batch dimension weighted by a uniform random vector from the n simplex
          (convex hull of unit vectors)
        """
        # compute the weights for the combination
        weights = random_weighting(n_blended)
        weights *= float(1 / np.linalg.norm(weights))
        weights = np.array(weights, dtype=np.double).reshape(-1, 1)
        # sum along the 0th dimension weighted by weights
        x = tf.tensordot(weights, tf.cast(x, tf.double), (0, 0))[0]
        y = tf.tensordot(weights, tf.cast(y, tf.double), (0, 0))[0]
        # return the convex combination
        return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    # map the dataset with the blend function
    dataset = dataset.map(lambda x, y: tf.py_function(blend, inp=[x, y], Tout=(tf.float32, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


def foff(x, y, alpha=.3, delta=3):
    """
    FMix data augmentation strategy function from https://arxiv.org/abs/2002.12047.  Applies the FMix data augmentation
    strategy to x and y with parameters alpha and delta described in the paper.  Not to be used as a dataset function,
    this is just the helper.

    :param x: Input image
    :param y: label (one hot encoded)
    :param alpha: alpha parameter from the FMix paper
    :param delta: delta parameter from the FMix paper
    :return: a mixed instance (x, y)
    """
    shape = (x.shape[-3], x.shape[-2])

    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, delta, x.shape[-1], *shape)
    spectrum = tf.dtypes.complex(spectrum[:, 0], spectrum[:, 1])
    mask = np.real(np.fft.irfftn(spectrum, shape))
    mask = mask[:1, :shape[0], :shape[1]]
    flat_mask = tf.reshape(mask, (shape[0] * shape[1],))
    top_k, _ = tf.math.top_k(flat_mask, k=int(alpha * shape[0] * shape[1]))
    kth = tf.reduce_min(top_k)

    mask = tf.reshape(tf.where(mask >= kth, tf.ones_like(mask), tf.zeros_like(mask)), (1, shape[0], shape[1]))

    mask = np.moveaxis(mask, 0, -1)
    mask = tf.stack([mask for i in range(x.shape[1])])
    mask = tf.stack([mask, 1 - mask])

    weights = np.array([alpha, 1 - alpha])

    x = tf.multiply(mask, tf.cast(x, tf.double))

    x = tf.reduce_sum(x, axis=0)
    y = tf.tensordot(tf.cast(weights, tf.double), tf.cast(y, tf.double), (0, 0))

    x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    return x, y


def fuzzy_fmix(x, y, delta=3):
    """
    'Fuzzy' version of the FMix data augmentation strategy function from https://arxiv.org/abs/2002.12047.  Applies the
    FMix data augmentation strategy to x and y without binarizing the fourier mask.  Parameters alpha and delta
    described in the paper.  Not to be used as a dataset function, this is just the helper.

    :param x: Input image
    :param y: label (one hot encoded)
    :param delta: delta parameter from the FMix paper
    :return: a mixed instance (x, y)
    """
    shape = (x.shape[-3], x.shape[-2])

    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, delta, x.shape[-1], *shape)
    spectrum = tf.dtypes.complex(spectrum[:, 0], spectrum[:, 1])
    mask = np.real(np.fft.irfftn(spectrum, shape))
    mask = mask[:1, :shape[0], :shape[1]]

    mask = (mask - tf.reduce_min(mask))
    mask = mask / tf.reduce_max(mask)

    mask = np.moveaxis(mask, 0, -1)
    mask = tf.stack([mask for i in range(x.shape[1])])
    mask = tf.stack([mask, 1 - mask])

    weights = np.array([tf.reduce_mean(mask), tf.reduce_mean(1 - mask)])

    x = tf.multiply(mask, tf.cast(x, tf.double))

    x = tf.reduce_sum(x, axis=0)
    # multiply weights along axis=0 and sum along the same axis
    y = tf.tensordot(weights, tf.cast(y, tf.double), (0, 0))

    x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    return x, y


def fmix_dset(train_dset, alpha=0.3, delta=3, **kwargs):
    """
    FMix data augmentation strategy function from https://arxiv.org/abs/2002.12047.  Applies the FMix data augmentation
    strategy to x and y with parameters alpha and delta described in the paper.

    :param alpha: alpha parameter from the FMix paper
    :param delta: delta parameter from the FMix paper
    :return: an augmented dataset
    """

    dataset = train_dset.batch(2)
    dataset = dataset.map(lambda x, y: tf.py_function(foff, inp=[x, y, tf.cast(alpha, tf.double),
                                                                    tf.cast(delta, tf.double)],
                                                         Tout=(tf.float32, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


def fast_fourier_fuckup_dset(train_ds, **kwargs):
    """
    'Fuzzy' version of the FMix data augmentation strategy function from https://arxiv.org/abs/2002.12047.  Applies the
    FMix data augmentation strategy to x and y without binarizing the fourier mask.  Parameters alpha and delta
    described in the paper.

    :param train_ds: dataset of batches to train on
    :return: an augmented dataset
    """

    dataset = train_ds.batch(2)
    dataset = dataset.map(lambda x, y: tf.py_function(fuzzy_fmix, inp=[x, y], Tout=(tf.float32, tf.float32)),
                          num_parallel_calls=tf.data.AUTOTUNE)

    return dataset
