"""
Utility functions for model training and result storage
"""


import os
import pickle
from itertools import product

from time import time, sleep
from random import randint

from warnings import warn

import tensorflow as tf
import numpy as np

import pynvml


class Config:
    """
    Configuration object for experiments.  Stores all of the information necessary to perform a deep learning experiment
    using this library.  Config consists of five dictionaries of parameters described below.

    :param hardware_params: only relevant for OSCER users, or potentially other SLURM based systems.  Includes the
        hardware parameters for the experiment.
        hardware_params must include:
            'n_gpu': uint - number of gpus to use for the experiment training
            'n_cpu': uint - number of cpu threads to use for training
            'node': str - name of compute node if applicable
            'partition': str - name of resource partition if applicable
            'time': str (we will just write this to the file) - max training time for slurm request
            'memory': uint - number of megabytes of RAM to allocate for SLURM request
            'results_dir': str - results directory local to ../
    :param network_params: parameters for building the network for training
        network_params must include:
            'network_fn': callable - network building function
            'network_args': dict - arguments to pass to network building function
                network_args must include:
                    'lrate': float - learning rate
            'hyperband': bool - whether or not to use hyperband hyperparameter search algorithm
    :param dataset_params: dataset parameter for training the network
        dataset_params must include:
            'dset_fn': callable - dataset loading function
            'dset_args': dict - arguments for dataset loading function
            'cache': str or bool - whether or not to cache the dataset (tf dataset caching has a memory leak)
            'batch': uint - batch size
            'prefetch': uint - prefetched batches
            'shuffle': bool - whether or not to shuffle after each epoch
            'augs': iterable - data augmentation functions
    :param experiment_params: miscellaneous experiment parameters for reproduceability and optimization
        experiment_params must include:
            'seed': int - random seed for computation
            'steps_per_epoch': uint - steps per epoch
            'patience': uint - patience for early stopping
            'min_delta': float - min delta for early stopping metric
            'epochs': uint - maximum number of training epochs
            'nogo': bool - if True model will not be trained
    :param optimization_params: parameter for model optimization
        optimization_params must include:
            'callbacks':  iterable - callbacks for training
            'training_loop': callable - training loop to use for model weight update

    methods:
        dump - dump config to a file
            :param fp: file pointer
        load - load fp from a file
            :param fp: file pointer
    """
    # this is bad practice I think, TODO: remove
    hardware_params = None
    network_params = None
    dataset_params = None
    experiment_params = None
    optimization_params = None

    def __init__(self, hardware_params=None, network_params=None, dataset_params=None, experiment_params=None,
                 optimization_params=None):
        self.hardware_params = hardware_params
        self.network_params = network_params
        self.dataset_params = dataset_params
        self.experiment_params = experiment_params
        self.optimization_params = optimization_params

    def dump(self, fp):
        pickle.dump(self, fp)

    def load(self, fp):
        obj = pickle.load(fp)
        self.hardware_params = obj.hardware_params
        self.network_params = obj.network_params
        self.dataset_params = obj.dataset_params
        self.experiment_params = obj.experiment_params
        self.optimization_params = obj.optimization_params
        if self.optimization_params['callbacks']:
            print("Note that callbacks cannot be serialized, so string representations were serialized instead")


class Results:
    """
    Convenience structure for storing results of training.  This is what will be pickled and stored in the results
    directory.  This structure contains all information required to summarize and analyze the experiment.

    :param experiment: Experiment object after training
    :param model_data: ModelData object after training
    """

    def __init__(self, experiment, model_data):
        self.config = Config(
                             experiment.hardware_params,
                             experiment.network_params,
                             experiment.dataset_params,
                             experiment.params,
                             experiment.optimization_params
                             )
        self.experiment = experiment
        self.model_data = model_data

        self.config.optimization_params['callbacks'] = [str(callback) for callback in self.config.optimization_params['callbacks']]
        self.experiment.optimization_params['callbacks'] = [str(callback) for callback in self.experiment.optimization_params['callbacks']]

        if self.config.optimization_params['callbacks'] or self.experiment.optimization_params['callbacks']:
            print("Note that callbacks cannot be serialized, so string representations will be serialized instead")

    def summary(self):
        """summarize the model training run"""
        metrics = [key for key in self.model_data.history]
        patience = self.config.experiment_params['patience']
        epochs = len(self.model_data.history['loss'])
        performance_at_patience = {key: self.model_data.history[key][epochs - patience - 1]
                                   for key in metrics}

        index = 'n/a' if self.experiment.index is None else self.experiment.index
        run_params = dict_to_string(dict(self.experiment.run_args)) if isinstance(self.experiment.run_args,
                                                                                  dict) else None
        print(f"""
------------------------------------------------------------
Experimental Results Summary (Index: {index})
------------------------------------------------------------
Dataset Params: {dict_to_string(self.experiment.dataset_params)}

Network Params:  {dict_to_string(self.experiment.network_params)}
------------------------------------------------------------
Experiment Parameters: {dict_to_string(self.experiment.params)}

Experiment Runtime: {self.model_data.run_time}s

Epochs Run / Average Time: {epochs} / {self.model_data.run_time / len(self.model_data.history['loss'])}s

Performance At Patience: {dict_to_string(performance_at_patience)}
------------------------------------------------------------
Runtime Parameters: {run_params}
------------------------------------------------------------
        """)


def augment_args(index, network_params):
    """
    Use the jobiterator to override the specified arguments based on the experiment index.

    Modifies the args

    :param args: arguments from ArgumentParser
    :return: A string representing the selection of parameters to be used in the file name
    """

    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    p = network_params['network_args']
    p['network_fn'] = network_params['network_fn']

    for key in p:
        if not isinstance(p[key], list):
            p[key] = [p[key]]

    # Check index number
    if index is None:
        return ""
    # Create the iterator
    ji = JobIterator({key: p[key] for key in set(p) - set(p['search_space']) - {'search_space'}}) \
        if network_params['hyperband'] else JobIterator({key: p[key] for key in set(p) - {'search_space'}})

    print("Size of Hyperparameter Grid:", ji.get_njobs())

    # Check bounds
    assert (0 <= index < ji.get_njobs()), "exp out of range"

    # Push the attributes to the args object and return a string that describes these structures
    augmented, arg_str = ji.set_attributes_by_index(index, network_params)

    if network_params['hyperband']:
        vars(augmented).update({key: p[key] for key in p['search_space']})
        vars(augmented).update({'search_space': p['search_space']})

    augmented = {
        'network_fn': augmented['network_fn'],
        'network_args': augmented,
        'hyperband': network_params['hyperband']
    }
    augmented['network_args'].pop('network_fn')
    augmented['network_args'].pop('network_args')

    return augmented, arg_str


def load_most_recent_results_with_fnames(d, n=1):
    """
    load the n most recent result files from the result directory
    :param d: directory from which to load the files
    :param n: number of recent results to load
    :return: a list of Results objects
    """
    results = []
    fnames = []

    for file in sorted(os.listdir(d), key=lambda f: os.stat(os.path.join(d, f)).st_mtime, reverse=True):
        if os.path.isfile(os.path.join(d, file)):
            with open(os.path.join(d, file), 'rb') as fp:
                results.append(pickle.load(fp))
                fnames.append(fp)
                n -= 1
                if not n:
                    break

    return results, fnames


def dict_to_string(d: dict, prefix="\t"):
    """
    helper method for pretty printing dictionaries.
    print each key/value pair from the dict on a new line

    :param d: (dict) a dictionary
    :param prefix: (string) prefix to append to delineate different levels
                   of dict hierarchy
    """
    s = "{\n"
    for k in d:
        if isinstance(d[k], dict):
            newfix = prefix + '\t'
            s += f"{prefix}{k}: {dict_to_string(d[k], newfix)}\n"
        else:
            s += f"{prefix}{k}: {d[k]}\n"
    return s + prefix + "}"


def load_most_recent_results(d, n=1):
    """
    load the n most recent result files from the result directory
    :param d: directory from which to load the files
    :param n: number of recent results to load
    :return: a list of Results objects
    """
    results = []

    for file in sorted(os.listdir(d), key=lambda f: os.stat(os.path.join(d, f)).st_mtime, reverse=True):
        if os.path.isfile(os.path.join(d, file)):
            with open(os.path.join(d, file), 'rb') as fp:
                results.append(pickle.load(fp))
                n -= 1
                if not n:
                    break

    return results


class JobIterator():
    """
    JobIterator object used to define the order of an array of experiments. By Andrew H. Fagg,
    modified by Jay Rothenberger
    """
    def __init__(self, params):
        """
        Constructor

        @param params Dictionary of key/list pairs
        """
        self.params = params
        # List of all combinations of parameter values
        self.product = list(dict(zip(params, x)) for x in product(*params.values()))
        # Iterator over the combinations
        self.iter = (dict(zip(params, x)) for x in product(*params.values()))

    def next(self):
        """
        @return The next combination in the list
        """
        return self.iter.next()

    def get_index(self, i):
        """
        Return the ith combination of parameters

        @param i Index into the Cartesian product list
        @return The ith combination of parameters
        """

        return self.product[i]

    def get_njobs(self):
        """
        @return The total number of combinations
        """

        return len(self.product)

    def set_attributes_by_index(self, i, obj):
        """
        For an arbitrary object, set the attributes to match the ith job parameters

        @param i Index into the Cartesian product list
        @param obj Arbitrary object (to be modified)
        @return A string representing the combinations of parameters, and the args object
        """

        # Fetch the ith combination of parameter values
        d = self.get_index(i)
        # Iterate over the parameters
        for k, v in d.items():
            obj[k] = v

        return obj, self.get_param_str(i)

    def get_param_str(self, i):
        """
        Return the string that describes the ith job parameters.
        Useful for generating file names

        @param i Index into the Cartesian product list
        """

        out = 'JI_'
        # Fetch the ith combination of parameter values
        d = self.get_index(i)
        # Iterate over the parameters
        for k, v in d.items():
            out = out + "%s_%s_" % (k, v)

        # Return all but the last character
        return out[:-1]


def prep_gpu(cpus_per_task, gpus_per_task=0, wait=True):
    """
    prepare the GPU for tensorflow computation

    :param cpus_per_task: number of threads to use for training
    :param gpus_per_task: number of gpu devices to use for training
    :param wait: if True wait from 0 to 5 minutes before allocating gpu memory
    """
    # initialize the nvidia management library
    pynvml.nvmlInit()
    # if we are not to use the gpu, then disable them
    n_physical_devices = gpus_per_task
    print(pynvml.nvmlDeviceGetCount(), tf.config.get_visible_devices('GPU'))
    tf.autograph.set_verbosity(0)
    if not gpus_per_task:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        n_physical_devices = 0
    else:
        try:
            # gpu handles
            gpus = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
            # -1 means "use every gpu"
            gpus_per_task = gpus_per_task if gpus_per_task > -1 else len(gpus)
            # get the fraction of used memory for each gpu
            if wait:
                sleep(randint(0, 60 * 5))
            usage = [pynvml.nvmlDeviceGetMemoryInfo(gpu).used / pynvml.nvmlDeviceGetMemoryInfo(gpu).total for gpu in
                     gpus]
            # sort the gpus by their available memory and filter out all gpus with more than 10% used
            avail = [i for i, v in sorted(list(enumerate(usage)), key=lambda k: k[-1], reverse=False) if v <= 1]
            # if we cannot satisfy the requested number of gpus this is an error
            if gpus_per_task > len(gpus):
                raise ValueError("too many gpus requested for this machine")
            # get the physical devices from tensorflow
            physical_devices = tf.config.get_visible_devices('GPU')
            # only take the number of gpus we plan to use
            avail_physical_devices = [physical_devices[i] for i in avail][:gpus_per_task]
            # set the visible devices only to the <gpus_per_task> least utilized
            tf.config.set_visible_devices(avail_physical_devices, 'GPU')
            n_physical_devices = len(avail_physical_devices)
        except Exception as e:
            warn(e)

    # use the available cpus to set the parallelism level
    if cpus_per_task is not None:
        pass
        # tf.config.threading.set_inter_op_parallelism_threads(cpus_per_task)
        # tf.config.threading.set_intra_op_parallelism_threads(cpus_per_task)

    if n_physical_devices > 1:
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)
        print('We have %d GPUs\n' % n_physical_devices)
    elif n_physical_devices:
        print('We have %d GPUs\n' % n_physical_devices)
    else:
        print('NO GPU')


def generate_fname(args):
    """
    Generate the base file name for output files/directories.

    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.

    :param args: from argParse
    :params_str: String generated by the JobIterator
    :return: a string (file name prefix) - just the last 6 digits of timestamp
    """
    return f"{str(time()).replace('.', '')[-6:]}"


class Experiment:
    """
        The Experiment class is used to run and enqueue deep learning jobs with a config dict

        :param config: a Config object
    """

    def __init__(self, config):
        self.index = None
        self.batch_file = None
        self.network_params = config.network_params
        self.hardware_params = config.hardware_params
        self.optimization_params = config.optimization_params

        if self.hardware_params.get('results_dir') is None:
            self.hardware_params['results_dir'] = 'results'

        self.dataset_params = config.dataset_params
        self.params = config.experiment_params
        self.run_args = None

    def run(self):
        """
        1. prep the hardware (prep_gpu)
        2. get the data
        3. make the model
        4. fit the model
        5. save the data
        """
        print(dict_to_string(self.hardware_params))
        print(dict_to_string(self.params))
        print(dict_to_string(self.network_params))
        print(dict_to_string(self.dataset_params))
        # set seed
        tf.random.set_seed(self.params['seed'])
        np.random.seed(self.params['seed'])

        prep_gpu(self.hardware_params['n_cpu'], self.hardware_params['n_gpu'], False)

        network_fn = self.network_params['network_fn']
        network_args = self.network_params['network_args']

        if self.hardware_params['n_gpu'] > 1:
            # create the scope
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                # build the model (in the scope)
                model = network_fn(**network_args)
        else:
            model = network_fn(**network_args)

        dset_dict = self.dataset_params['dset_fn'](**self.dataset_params['dset_args'])

        train_dset, val_dset, test_dset = dset_dict['train'], dset_dict['val'], dset_dict['test']

        def postprocess_dset(ds):
            ds = ds.repeat()

            if self.dataset_params['batch'] > 1:
                ds = ds.batch(self.dataset_params['batch'])

            if self.dataset_params['cache']:
                if self.dataset_params['cache_to_lscratch']:
                    ds = ds.cache(self.run_args.lscratch)
                else:
                    ds = ds.cache()

            if self.dataset_params['shuffle']:
                ds = ds.shuffle(self.dataset_params['shuffle'], self.params['seed'], True)

            ds = ds.prefetch(self.dataset_params['prefetch'])

            return ds

        val_dset = val_dset.batch(self.dataset_params['batch'])

        train_dset = postprocess_dset(train_dset)

        for aug in self.dataset_params['augs']:
            train_dset = aug(train_dset)
        # train the model
        if self.network_params['hyperband']:
            return self.optimization_params['training_loop'](model,
                                                             train_dset,
                                                             val_dset,
                                                             self.network_params,
                                                             self.params,
                                                             self.optimization_params['callbacks'],
                                                             train_steps=self.params['steps_per_epoch'],
                                                             evaluate_on={'test': test_dset})

        model_data = self.optimization_params['training_loop'](model, train_dset, val_dset, self.network_params,
                                                               self.params,
                                                               self.optimization_params['callbacks'],
                                                               train_steps=self.params['steps_per_epoch'])

        result = Results(self, model_data)
        with open(f'{os.curdir}/../{self.hardware_params["results_dir"]}/{generate_fname(self.params)}', 'wb') as fp:
            pickle.dump(result, fp)

    def run_array(self, index=0):
        self.index = index
        self.network_params, _ = augment_args(index, self.network_params)
        self.run()

    def enqueue(self):
        """
        1. save the current Experiment object to a pickle temp file
        2. create the batch file / hardware params
        3. sbatch the batch file
        """
        exp_file = f"experiments/experiment-{str(time()).replace('.', '')}.pkl"
        batch_text = f"""#!/bin/bash
#SBATCH --partition={self.hardware_params['partition']}
#SBATCH --cpus-per-task={self.hardware_params['n_cpu']}
#SBATCH --ntasks=1
#SBATCH --mem={self.hardware_params['memory']}
#SBATCH --output={self.hardware_params['stdout_path']}
#SBATCH --error={self.hardware_params['stderr_path']}
#SBATCH --time={self.hardware_params['time']}
#SBATCH --job-name={self.hardware_params['name']}
#SBATCH --mail-user={self.hardware_params['email']}
#SBATCH --mail-type=ALL
#SBATCH --chdir={self.hardware_params['dir']}
#SBATCH --nodelist={','.join(self.hardware_params['nodelist'])}
#SBATCH --array={self.hardware_params['array']}
. /home/fagg/tf_setup.sh
conda activate tf
python run.py --pkl {exp_file} --lscratch $LSCRATCH --id $SLURM_ARRAY_TASK_ID"""

        with open('experiment.sh', 'w') as fp:
            fp.write(batch_text)

        with open(exp_file, 'wb') as fp:
            pickle.dump(self, fp)

        self.batch_file = batch_text

        os.system(f'sbatch experiment.sh')
