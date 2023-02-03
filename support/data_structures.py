"""
Custom data structures to support model training, evaluation, and result storage
"""

import pickle

import numpy as np
from dataclasses import dataclass

from sklearn.metrics import confusion_matrix


@dataclass
class ModelData:
    """
    This object is meant to hold all of the data neccessary to reconstruct the model after training.

    :param weights:  Neural network weights array (initially empty)
    :param network_params:  Parameters passed to the network_fn to build the network stored in this object
    :param network_fn:  Function that builds this network given the parameters (weights need not be initialized)
    :param evaluations:  Dataset evaluation results (usually train, val, test metrics).  Initially empty.
    :param classes:  Number of classes (for image classification)
    :param history:  Training epoch history.  Metrics for each epoch of training.
    :param run_time:  Run time of model training in seconds.
    """
    weights: np.ndarray
    network_params: dict
    network_fn: callable
    evaluations: dict
    classes: list
    history: dict
    run_time: float

    def report(self):
        # give a performance report for an individual model
        for key in self.evaluations:
            print(f'{key}:')
            print(self.evaluations[key])

    def confusion_matrix(self, dset, steps=None):
        # generate an ndarray confusion matrix
        # accepts a non-shuffling generator with optional steps
        steps = len(dset) if steps is None else steps
        pred = np.argmax(self.get_model().predict(dset, steps=steps))
        return confusion_matrix(dset.classes, pred, labels=np.unique(dset.classes))

    def get_history(self):
        # return the history object for the model
        return self.history

    def get_model(self):
        # return the keras model
        model = self.network_fn(**self.network_params['network_args'])
        model.set_weights(self.weights)

        return model


@dataclass
class ModelEvaluator:
    """
    Model Evaluator object used to select best hyperparameters.

    :param models: a list of ModelData objects
    """
    models: list

    def best_hyperparams(self, train_fraction=None, metric='loss', mode=min, split='val'):
        """
        finds and returns a dictionary of the best hyperparameters over the model evaluations in the models list, and
        the performance metrics for the corresponding model

        :param train_fraction: fraction of training data to scan over (float, iterable of floats, or None) if None looks over all training fractions
        :param metric: metric over which to optimize
        :param mode: function to define what 'best' means, reasonable choices include: min, max
        :param split: split to calculate best over
        :return: a dict of the best hyperparameter name: value pairs
        """

        if isinstance(train_fraction, float):
            train_fraction = [train_fraction]
        elif isinstance(train_fraction, int):
            train_fraction = [train_fraction]
        elif train_fraction is None:
            train_fraction = self.unique_fractions()
        # validation loss minimizing parameters
        # this comprehension returns a ModelData instance
        # it contains the ModelData instance of the stored model and the metrics on which it was evaluated

        best = min([model
                    for model in self.models if model.train_fraction in train_fraction],
                   key=lambda x: x.evaluations[split][metric])

        return best

    def to_pickle(self, filename):
        """pickle the object and save it to filename"""
        with open(filename, 'wb') as fp:
            pickle.dump(self, fp)

    def unique_fractions(self):
        """find the unique training fraction amounts"""
        rax = set()
        for model in self.models:
            rax.add(model.train_fraction)

        return rax

    def append(self, model: ModelData):
        """append a model to the internal models list"""
        self.models.append(model)
