import argparse

import numpy as np
from supervised.util import load_most_recent_results_with_fnames as load_most_recent_results
from itertools import product
from supervised.util import dict_to_string, prep_gpu

import matplotlib.pyplot as plt
import seaborn as sn


# form a numpy tensor that grids a list of hyperparameters

def create_parser():
    """
    Create argument parser
    """
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='CNN', fromfile_prefix_chars='@')
    # what kind of explanations to generate
    parser.add_argument('--dir', type=str, default=None, help="experiment results directory")
    parser.add_argument('--shape', type=int, nargs='+', help="shape of the hyperparameter grid")
    parser.add_argument('--hparams', type=str, nargs='+', help="string names of hyperparameters in the grid")
    parser.add_argument('--metric', type=str, default=None, help="metric to plot as the elements of the grid")

    return parser


if __name__ == "__main__":
    prep_gpu(12, 1, False)
    parser = create_parser()
    args = parser.parse_args()
    print(int(np.prod(args.shape)))
    most_recent, fnames = load_most_recent_results(args.dir, np.prod(args.shape))
    grid = np.zeros(tuple(args.shape))
    fnames_grid = np.zeros_like(grid).astype(np.str)
    print(fnames)
    fnames = [str(fname).split("'")[-2].split('/')[-1] for fname in fnames]

    values = {hpar: set() for hpar in args.hparams}
    # query the values along the axis
    for results in most_recent:
        for hpar in args.hparams:
            values[hpar].add(results.config.network_params['network_args'][hpar])

    values = {hpar: sorted(list(values[hpar])) for hpar in args.hparams}
    # put the metric in the appropriate position for each model
    for results, fname in zip(most_recent, fnames):
        result_coord = [0 for hpar in args.hparams]

        for i, hpar in enumerate(args.hparams):
            result_coord[i] = values[hpar].index(results.config.network_params['network_args'][hpar])

        grid[tuple(result_coord)] = results.model_data.history[args.metric][len(results.model_data.history['loss']) -
                                                                            results.config.experiment_params['patience']
                                                                            - 1]
        fnames_grid[tuple(result_coord)] = fname

    print(dict_to_string(values))

    print(np.flip(grid, 1))
    print(np.flip(fnames_grid, 1))

    sn.heatmap(np.flip(grid, 1), cbar=False, annot=True, xticklabels=values[args.hparams[1]][::-1],
               yticklabels=values[args.hparams[0]])
    plt.xlabel(args.hparams[1])
    plt.ylabel(args.hparams[0])
    plt.title(args.metric + ' heatmap')
    plt.show()
