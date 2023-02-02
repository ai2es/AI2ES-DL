import tensorflow as tf
import tensorflow_datasets as tfds


def diffusion_dataset():
    # TODO
    pass


def cifar10(image_size=(32, 32, 3), center=True, **kwargs):
    ds = tfds.load('cifar10', split='train', as_supervised=True)
    val = tfds.load('cifar10', split='train[80%:]', as_supervised=True)
    test = tfds.load('cifar10', split='test', as_supervised=True)

    def to_unsupervised(dataset):
        images, labels = tuple(zip(*dataset))
        dataset = tf.data.Dataset.from_tensor_slices(tf.stack(images, 0))
        return dataset

    ds = to_unsupervised(ds)
    val = to_unsupervised(val)
    test = to_unsupervised(test)

    def preprocess_image(x):
        """
        Load in image from filename and resize to target shape.
        """
        image = tf.image.convert_image_dtype(x, tf.float32)
        image = tf.image.resize(image, (image_size[0], image_size[1]))
        if center:
            image = image - tf.reduce_mean(image)

        return image, image

    ds = ds.map(
        lambda x: tf.py_function(preprocess_image, inp=[x], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    val = val.map(
        lambda x: tf.py_function(preprocess_image, inp=[x], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    test = test.map(
        lambda x: tf.py_function(preprocess_image, inp=[x], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    return {'train': ds, 'val': val, 'test': test, 'class_names': [str(i) for i in range(10)]}


def cifar100(image_size=(32, 32, 3), center=True, **kwargs):
    ds = tfds.load('cifar100', split='train', as_supervised=True)
    val = tfds.load('cifar100', split='train[80%:]', as_supervised=True)
    test = tfds.load('cifar100', split='test', as_supervised=True)

    def to_unsupervised(dataset):
        images, labels = tuple(zip(*dataset))
        dataset = tf.data.Dataset.from_tensor_slices(tf.stack(images, 0))
        return dataset

    ds = to_unsupervised(ds)
    val = to_unsupervised(val)
    test = to_unsupervised(test)

    def preprocess_image(x):
        """
        Load in image from filename and resize to target shape.
        """
        image = tf.image.convert_image_dtype(x, tf.float32)
        image = tf.image.resize(image, (image_size[0], image_size[1]))
        if center:
            image = image - tf.reduce_mean(image)

        return image, image

    ds = ds.map(
        lambda x: tf.py_function(preprocess_image, inp=[x], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    val = val.map(
        lambda x: tf.py_function(preprocess_image, inp=[x], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    test = test.map(
        lambda x: tf.py_function(preprocess_image, inp=[x], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    return {'train': ds, 'val': val, 'test': test, 'class_names': [str(i) for i in range(100)]}
