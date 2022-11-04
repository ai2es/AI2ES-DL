import tensorflow as tf
import tensorflow_datasets as tfds

from supervised.datasets.util import df_from_dirlist, to_dataset

"""
data loading functions must only return a dictionary of this form:
{'train': train_dset, 'val': val_dset, 'test': test_dset}
where each _dset is a finite tf.data.Dataset object whose elements are single (unbatched) input-output pairs
"""


def cats_dogs(batch_size=16, image_size=(128, 128, 3), cache=False, prefetch=4, **kwargs):
    setattr(tfds.image_classification.cats_vs_dogs, '_URL',
            "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F"
            "/kagglecatsanddogs_5340.zip")
    ds = tfds.load('cats_vs_dogs', split='train[:70%]', as_supervised=True)
    val = tfds.load('cats_vs_dogs', split='train[70:80%]', as_supervised=True)
    test = tfds.load('cats_vs_dogs', split='train[80%:]', as_supervised=True)

    def preprocess_image(x, y, center=True):
        """
        Load in image from filename and resize to target shape.
        """
        image = tf.image.convert_image_dtype(x, tf.float32)
        image = tf.image.resize(image, (image_size[0], image_size[1]))
        if center:
            image = image - tf.reduce_mean(image)

        label = tf.one_hot(y, 2, dtype=tf.float32)

        return image, label

    ds = ds.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y, True], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    val = val.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y, True], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    test = test.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y, True], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    return {'train': ds, 'val': val, 'test': test, 'class_names': ['cats', 'dogs']}


def deep_weeds(image_size=(128, 128, 3), center=True, **kwargs):

    ds = tfds.load('deep_weeds', split='train[:70%]', as_supervised=True)
    val = tfds.load('deep_weeds', split='train[70:80%]', as_supervised=True)
    test = tfds.load('deep_weeds', split='train[80%:]', as_supervised=True)

    def preprocess_image(x, y):
        """
        Load in image from filename and resize to target shape.
        """
        image = tf.image.convert_image_dtype(x, tf.float32)
        image = tf.image.resize(image, (image_size[0], image_size[1]))
        if center:
            image = image - tf.reduce_mean(image)

        label = tf.one_hot(y, 9, dtype=tf.float32)

        return image, label

    ds = ds.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    val = val.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y, True], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    test = test.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    return {'train': ds, 'val': val, 'test': test, 'class_names': ['chinee', 'lantana', 'parkinsonia', 'parenthenium',
                                                                   'prickly', 'rubber', 'siam', 'snake', 'none']}


def citrus_leaves(image_size=(128, 128, 3), **kwargs):

    ds = tfds.load('citrus_leaves', split='train[:70%]', as_supervised=True)
    val = tfds.load('citrus_leaves', split='train[70:80%]', as_supervised=True)
    test = tfds.load('citrus_leaves', split='train[80%:]', as_supervised=True)

    def preprocess_image(x, y, center=True):
        """
        Load in image from filename and resize to target shape.
        """
        image = tf.image.convert_image_dtype(x, tf.float32)
        image = tf.image.resize(image, (image_size[0], image_size[1]))
        if center:
            image = image - tf.reduce_mean(image)

        label = tf.one_hot(y, 4, dtype=tf.float32)

        return image, label

    ds = ds.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    val = val.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y, True], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    test = test.map(
        lambda x, y: tf.py_function(preprocess_image, inp=[x, y], Tout=(tf.float32, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE)

    return {'train': ds, 'val': val, 'test': test, 'class_names': ['black spot', 'canker', 'greening', 'healthy']}


def dot_dataset(path, image_size=(256, 256, 3)):
    paths = ['bronx_allsites/wet', 'bronx_allsites/dry', 'bronx_allsites/snow',
             'ontario_allsites/wet', 'ontario_allsites/dry', 'ontario_allsites/snow',
             'rochester_allsites/wet', 'rochester_allsites/dry', 'rochester_allsites/snow']

    paths = [path + p for p in paths]

    df, class_map = df_from_dirlist(paths)

    df = df.sample(frac=1).reset_index(drop=True)

    ds, val, test = df.iloc[:round((len(df) / 10)*7)], df.iloc[round((len(df) / 10)*7):round((len(df) / 10)*8)],\
                    df.iloc[round((len(df) / 10)*8):]

    val = to_dataset(val, class_mode='categorical', image_size=image_size)
    test = to_dataset(test, class_mode='categorical', image_size=image_size)
    ds = to_dataset(ds, class_mode='categorical', image_size=image_size)

    return {'train': ds, 'val': val, 'test': test, 'class_names': ['dry', 'snow', 'wet']}
