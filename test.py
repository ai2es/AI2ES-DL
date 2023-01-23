from supervised.util import Config, Experiment, load_most_recent_results

from supervised.models.cnn import build_EfficientNetB0, build_camnetv2, build_camnet, build_basic_cnn,\
    build_camnet_reorderedv5, build_camnet_reorderedv4, build_focal_modulator, build_focal_camnet, build_focal_camnetv2

from supervised.datasets.image_classification import deep_weeds, cats_dogs, dot_dataset, citrus_leaves
from supervised.data_augmentation.msda import mixup_dset, blended_dset
from supervised.data_augmentation.ssda import add_gaussian_noise_dset, custom_rand_augment_dset, foff_dset
"""
hardware_params must include:

    'n_gpu': uint
    'n_cpu': uint
    'node': str
    'partition': str
    'time': str (we will just write this to the file)
    'memory': uint
    'distributed': bool
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
    'stdout_path': '/scratch/jroth/supercomputer/text_outputs/exp%01a_stdout_%A.txt',
    'stderr_path': '/scratch/jroth/supercomputer/text_outputs/exp%01a_stderr_%A.txt',
    'email': 'jay.c.rothenberger@ou.edu',
    'dir': '/scratch/jroth/AI2ES-DL/',
    'array': '[0-48%4]',
    'results_dir': 'results'
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
    'network_fn': build_camnet_reorderedv5,
    'network_args': {
        'lrate': 5e-4,
        'depth': 3,
        'n_classes': 9,
        'iterations': 6,
        'conv_filters': '[12]',
        'conv_size': '[3]',
        'dense_layers': '[32, 16]',
        'learning_rate': 5e-4,
        'image_size': image_size,
        'l1': None,
        'l2': None,
        'alpha': [2**(-10)],
        'beta': [.5],
        'noise_level': 0.005,
        'depth': 5,
    },
    'hyperband': False
}

"""
experiment_params must include:
    
    'seed': random seed for computation
    'steps_per_epoch': uint
    'validation_steps': uint
    'patience': uint
    'min_delta': float
    'epochs': uint
    'nogo': bool
"""
experiment_params = {
    'seed': 42,
    'steps_per_epoch': 512,
    'validation_steps': 256,
    'patience': 32,
    'min_delta': 0.0,
    'epochs': 256,
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
    'dset_fn': deep_weeds,
    'dset_args': {
        'image_size': image_size[:-1],
        'path': '../data/'
    },
    'cache': False,
    'cache_to_lscratch': False,
    'batch': 16,
    'prefetch': 4,
    'shuffle': True,
    'augs': []
}

config = Config(hardware_params, network_params, dataset_params, experiment_params)

if __name__ == "__main__":

    exp = Experiment(config)

    print(exp.params)
    exp.run_array(0)

    exp.enqueue()
