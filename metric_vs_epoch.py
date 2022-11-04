from supervised.evaluations.image_classification import show_mask, explain_image_classifier_with_saliency,\
    explain_image_classifier_with_shap, explain_image_classifier_with_lime

from supervised.util import load_most_recent_results, dict_to_string

import matplotlib.pyplot as plt
import numpy as np

import pickle
import argparse


def create_parser():
    """
    Create argument parser
    """
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='CNN', fromfile_prefix_chars='@')
    # what kind of explanations to generate
    parser.add_argument('--fname', type=str, default=None, help="experiment result file")
    parser.add_argument('--metric', type=str, default='loss', help="metric to plot vs epoch")

    return parser


def figure_metric_epoch(model, title, metric):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 16

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes

    plt.plot(range(len(model.history[metric])), model.history[metric])
    # add the plot readability information
    plt.title(title)

    plt.xlabel('epoch')
    plt.ylabel(metric)

    # save the figure
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.show()
    plt.clf()
    plt.close()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.fname, 'rb') as fp:
        results = pickle.load(fp)
    results.summary()
    figure_metric_epoch(results.model_data, f"{args.metric} v.s. epoch", args.metric)
