.. AI2ES Deep Learning documentation master file, created by
   sphinx-quickstart on Thu Feb  2 11:51:23 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AI2ES-DL
==================
example deep learning pipeline for OSCER written in Tensorflow

using the pipeline with your own models, datasets, and data augmentation strategies is very simple.  One only needs to follow a few conventions:

using your own models: (see trainable/models/cnn.py)

1. network functions should only take int, float, or string typed arguments

2. network functions must return only a compiled keras model

using your own datasets: (see data/datasets/image_classification.py)

1. dataset functions should return only an unbatched finite tf.data.Dataset object

using your own data augmentation strategies: (see optimization/data_augmentation/ssda.py)

1. data augmentation functions must require only as input a dataset
2. data augmentation functions must return only a tf.data.Dataset object representing the augmented dataset

see `test.py <https://github.com/ai2es/AI2ES-DL/blob/unstable/test.py>`_ for an example use case for supervised image classification

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   optimization
   optimization.training_loops
   optimization.data_augmentation
   data
   data.datasets
   support
   support.evaluations
   trainable
   trainable.models


