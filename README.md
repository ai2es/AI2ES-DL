# AI2ES-DL
example deep learning pipeline for OSCER written in Tensorflow

using the pipeline with your own models, datasets, and data augmentation strategies is very simple.  One only needs to follow a few conventions:

using your own models:

1. network functions should only take int, float, or string typed arguments
2. network functions must return only a compiled keras model

using your own datasets:
1. dataset functions should return only an unbatched finite tf.data.Dataset object

using your own data augmentation strategies:
1. data augmentation functions must require only as input a dataset
2. data augmentation functions must return only a tf.data.Dataset object representing the augmented dataset

see test.py for an example use case for supervised image classification


