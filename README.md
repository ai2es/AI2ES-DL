# AI2ES-DL
example deep learning pipeline for OSCER written in Tensorflow

using the pipeline with your own models, datasets, and data augmentation strategies is very simple.  One only needs to follow a few conventions:

using your own models: (see supervised/models/cnn.py)

1. network functions should only take int, float, or string typed arguments
2. network functions must return only a compiled keras model

using your own datasets: (see supervised/datasets/image_classification.py)
1. dataset functions should return only an unbatched finite tf.data.Dataset object

using your own data augmentation strategies: (see supervised/data_augmentation/ssda.py)
1. data augmentation functions must require only as input a dataset
2. data augmentation functions must return only a tf.data.Dataset object representing the augmented dataset

see test.py for an example use case for supervised image classification


