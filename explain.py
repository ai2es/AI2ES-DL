from supervised.evaluations.image_classification import show_mask, explain_image_classifier_with_saliency,\
    explain_image_classifier_with_shap, explain_image_classifier_with_lime

from supervised.util import load_most_recent_results

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
    parser.add_argument('--thrifty', action='store_true', help="thrifty i2i explanation (camnet)")
    parser.add_argument('--saliency', action='store_true', help="deconvolution saliency")
    parser.add_argument('--lime', action='store_true', help="LIME explanation")
    parser.add_argument('--shap', action='store_true', help="SHAP explanation")

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.fname, 'rb') as fp:
        results = pickle.load(fp)
    results.summary()
    results.config.dataset_params['dset_args']['path'] = '../Semi-supervised/data/'
    class_names = results.config.dataset_params['dset_fn'](**results.config.dataset_params['dset_args'])['class_names']
    model_data = results.model_data
    keras_model = model_data.get_model()
    test_dset = results.config.dataset_params['dset_fn'](**results.config.dataset_params['dset_args'])['test']
    test_dset = test_dset.batch(1)

    for x, y in iter(test_dset):
        print(x.shape, y.shape)
        break

    if args.thrifty:
        show_mask(test_dset, 3, model_data, class_names=class_names)

    if args.lime:
        for x, y in iter(test_dset):
            explain_image_classifier_with_lime(keras_model, x[0], len(class_names))
            break

    if args.shap:
        for x, y in iter(test_dset):
            explain_image_classifier_with_shap(keras_model, x, class_names)
            break

    if args.saliency:
        for x, y in iter(test_dset):
            explain_image_classifier_with_saliency(keras_model, x)
            break
