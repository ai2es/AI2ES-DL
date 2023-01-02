import tensorflow as tf
import numpy as np

HARDSWISH = lambda x: x * tf.nn.relu6(x + 3) / 6
MISH = lambda x: x * tf.nn.tanh(tf.nn.softplus(x))


def int_act_slip(x):
    # activation function that just pushes every real number closer to its closest integer (except x % 1 = .5)
    # this step function has slippery steps
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
    # activation function that just pushes every real number closer to its closest integer (except x % 1 = .5)
    # this step function has inwardly sloping steps
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


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -1)
    emb = tf.reshape(emb, (*emb.shape[:-2], -1))
    return emb


class TFPositionalEncoding2D(tf.keras.layers.Layer):
    def __init__(self, channels: int, dtype=tf.float32):
        """
        Args:
            channels int: The last dimension of the tensor you want to apply pos emb to.
        Keyword Args:
            dtype: output type of the encodings. Default is "tf.float32".
        """
        super(TFPositionalEncoding2D, self).__init__()

        self.channels = int(2 * np.ceil(channels / 4))
        self.inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, self.channels, 2) / np.float32(self.channels)
            )
        )
        self.cached_penc = None

    @tf.function
    def call(self, inputs):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(inputs.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == inputs.shape:
            return self.cached_penc

        self.cached_penc = None
        _, x, y, org_channels = inputs.shape

        dtype = self.inv_freq.dtype

        pos_x = tf.range(x, dtype=dtype)
        pos_y = tf.range(y, dtype=dtype)

        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = tf.einsum("i,j->ij", pos_y, self.inv_freq)

        emb_x = tf.expand_dims(get_emb(sin_inp_x), 1)
        emb_y = tf.expand_dims(get_emb(sin_inp_y), 0)

        emb_x = tf.tile(emb_x, (1, y, 1))
        emb_y = tf.tile(emb_y, (x, 1, 1))
        emb = tf.concat((emb_x, emb_y), -1)
        self.cached_penc = tf.repeat(
            emb[None, :, :, :org_channels], tf.shape(inputs)[0], axis=0
        )
        return self.cached_penc


class PCA(tf.keras.layers.Layer):
    """
    Layer that iteratively updates a PCA estimate and transforms the output
    into the PC space

    inputs: a batch of n-D vectors
    outputs: a batch of 1-D vectors in PC coordinate space
    """

    def Q_init(self, shape, dtype=tf.float32, **kwargs):
        H = tf.random_normal_initializer(0.0, 1.0, 42)(shape, dtype=dtype)
        Q, _ = tf.linalg.qr(H, full_matrices=False)
        return Q

    def streaming_tf_PCA(self, X, p, k):
        # the examples are vectors of size p, want to deconstruct them into features k
        # this means H is p x k
        X = tf.cast(X, tf.float32)
        X, _ = tf.linalg.normalize(X, axis=-1)
        B = tf.shape(X)[0]
        B = tf.convert_to_tensor(float(B), dtype=tf.float32)
        S = tf.zeros_like(self.Q, dtype=tf.float32)
        XXT = tf.linalg.matmul(tf.transpose(X), X) / B
        S = S + tf.linalg.matmul(XXT, self.Q)

        Q, R = tf.linalg.qr(S, full_matrices=False, name='name')

        self.Q.assign(Q)

    def __init__(self, num_outputs):
        super(PCA, self).__init__()
        self.num_outputs = num_outputs
        self.Q = None

    def build(self, input_shape):
        self.Q = tf.Variable(self.Q_init((tf.reduce_prod(input_shape)[1:], self.num_outputs)))

    def call(self, inputs, training=None):
        input_shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, (input_shape[0], tf.reduce_prod(input_shape[1:])))
        if training:
            self.streaming_tf_PCA(inputs, inputs.shape[-1], self.num_outputs)

        return tf.matmul(inputs, self.Q)

    def get_config(self):
        return {"Q": self.Q.numpy()}


class PCACompress(tf.keras.layers.Layer):
    """
    Layer that iteratively updates a PCA estimate and transforms the output
    into the PC space

    inputs: a batch of n-D vectors
    outputs: a batch of n-D vectors reconstructed from PCs
    """

    def Q_init(self, shape, dtype=tf.float32, **kwargs):
        H = tf.random_normal_initializer(0.0, 1.0, 42)(shape, dtype=dtype)
        Q, _ = tf.linalg.qr(H, full_matrices=False)
        return Q

    def streaming_tf_PCA(self, X, p, k):
        # the examples are vectors of size p, want to deconstruct them into features k
        # this means H is p x k
        X = tf.cast(X, tf.float32)
        X, _ = tf.linalg.normalize(X, axis=-1)
        B = tf.shape(X)[0]
        B = tf.convert_to_tensor(float(B), dtype=tf.float32)
        S = tf.zeros_like(self.Q, dtype=tf.float32)
        XXT = tf.linalg.matmul(tf.transpose(X), X) / B
        S = S + tf.linalg.matmul(XXT, self.Q)

        Q, R = tf.linalg.qr(S, full_matrices=False, name='name')

        self.Q.assign(Q)

    def __init__(self, num_outputs):
        super(PCA_Compress, self).__init__()
        self.num_outputs = num_outputs
        self.Q = None

    def build(self, input_shape):
        input_shape = tf.reduce_prod(input_shape[1:])
        self.Q = tf.Variable(self.Q_init((input_shape, self.num_outputs)))

    def call(self, inputs, training=None):
        input_shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, (input_shape[0], tf.reduce_prod(input_shape[1:])))
        if training:
            self.streaming_tf_PCA(inputs, inputs.shape[-1], self.num_outputs)

        return tf.reshape(tf.linalg.matmul(tf.linalg.matmul(inputs, self.Q), tf.transpose(self.Q)), input_shape)

    def get_config(self):
        return {"Q": self.Q.numpy()}
    


