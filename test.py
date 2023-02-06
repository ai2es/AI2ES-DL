import tensorflow as tf

from support.util import Config, Experiment

from trainable.models.vit import build_focal_LAXNet

from data.datasets.image_classification import cats_dogs
from optimization.data_augmentation.ssda import custom_rand_augment_dset

from tensorflow.keras.callbacks import LearningRateScheduler
from optimization.callbacks import EarlyStoppingDifference

from optimization.training_loops.supervised import keras_supervised
from optimization.schedules import bleed_out
"""
hardware_params must include:

    'n_gpu': uint
    'n_cpu': uint
    'node': str
    'partition': str
    'time': str (we will just write this to the file)
    'memory': uint
    'results_dir': str
"""
hardware_params = {
    'name': 'hparam',
    'n_gpu': 1,
    'n_cpu': 16,
    'partition': 'ai2es',
    'nodelist': ['c830'],
    'time': '96:00:00',
    'memory': 16384,
    # The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
    'stdout_path': '/ourdisk/hpc/ai2es/jroth/supercomputer/text_outputs/exp%01a_stdout_%A.txt',
    'stderr_path': '/ourdisk/hpc/ai2es/jroth/supercomputer/text_outputs/exp%01a_stderr_%A.txt',
    'email': 'jay.c.rothenberger@ou.edu',
    'dir': '/ourdisk/hpc/ai2es/jroth/AI2ES_DL/',
    'array': '[0-7%4]',
    'results_dir': 'cats_dogs_grid'
}
"""
network_params must include:
    
    'network_fn': network building function
    'network_args': arguments to pass to network building function
        network_args must include:
            'lrate': float
    'hyperband': bool
"""
image_size = (128, 128, 3)

network_params = {
    'network_fn': build_focal_LAXNet,
    'network_args': {
        'n_classes': 2,
        'conv_filters': 24,
        'conv_size': '[3]',
        'dense_layers': '[16]',
        'learning_rate': 5e-4,
        'image_size': image_size,
        'l1': None,
        'l2': None,
        'alpha': [1, 2**(-1), 2**(-3), 2**(-7), 2**(-10), 2**(-13), 0],
        'beta': [1, 2**(-1), 2**(-3), 2**(-7), 2**(-10), 2**(-13), 0],
        'gamma': 0,
        'noise_level': 0.005,
        'depth': 5,
    },
    'hyperband': False
}

"""
experiment_params must include:
    
    'seed': random seed for computation
    'steps_per_epoch': uint
    'patience': uint
    'min_delta': float
    'epochs': uint
    'nogo': bool
"""


experiment_params = {
    'seed': 42,
    'steps_per_epoch': 512,
    'patience': 32,
    'min_delta': 0.0,
    'epochs': 512,
    'nogo': False,
}
"""
dataset_params must include:
    'dset_fn': dataset loading function
    'dset_args': arguments for dataset loading function
    'cache': str or bool
    'batch': uint
    'prefetch': uint
    'shuffle': bool
    'augs': iterable of data augmentation functions
"""
dataset_params = {
    'dset_fn': cats_dogs,
    'dset_args': {
        'image_size': image_size[:-1],
        'path': '../data/'
    },
    'cache': False,
    'cache_to_lscratch': False,
    'batch': 97,
    'prefetch': 8,
    'shuffle': True,
    'augs': []
}

"""
optimization_params must include:
    'callbacks':  iterable - callbacks for training
    'training_loop': callable - training loop to use for model weight update
"""

optimization_params = {
    'callbacks': [
        # EarlyStoppingDifference(patience=experiment_params['patience'],
        #                        restore_best_weights=True,
        #                        min_delta=experiment_params['min_delta'],
        #                        metric_0='val_clam_categorical_accuracy',
        #                        metric_1='val_clam_1_categorical_accuracy',
        #                        n_classes=2),
        tf.keras.callbacks.EarlyStopping(
                                         patience=experiment_params['patience'],
                                         restore_best_weights=True,
                                         min_delta=experiment_params['min_delta'],
                                         monitor='val_clam_categorical_accuracy'
                                        ),
        LearningRateScheduler(bleed_out(network_params['network_args']['learning_rate'])),
        # LossWeightScheduler(loss_weight_schedule)
    ],
    'training_loop': keras_supervised
}

config = Config(hardware_params, network_params, dataset_params, experiment_params, optimization_params)


if __name__ == "__main__":

    exp = Experiment(config)

    # print(exp.params)
    # exp.run_array(0)

    exp.enqueue()