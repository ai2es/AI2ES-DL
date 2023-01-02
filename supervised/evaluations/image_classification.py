from time import time

import matplotlib.pyplot as plt
import numpy as np

import os

from lime import lime_image
import shap

from skimage.io import imread
from skimage.segmentation import mark_boundaries

import tensorflow as tf


def explain_image_classifier_with_lime(model, instance, n_classes):
    """
    show a visual explanation using LIME for an image classification keras Model and a image instance with matplotlib
    """
    instance = np.array(instance)
    explainer = lime_image.LimeImageExplainer(kernel_width=.125)
    explanation = explainer.explain_instance(instance.astype(np.double), model.predict, top_labels=n_classes,
                                             hide_color=0, num_samples=2048, batch_size=16)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=True)
    plt.imshow(mark_boundaries(temp / 2 + .5, mask))
    plt.show()


def explain_image_classifier_with_shap(model, instance, class_names):
    instance = np.array(instance).astype(np.double)
    print(instance.shape)
    # define a masker that is used to mask out partitions of the input image, this one uses a blurred background
    masker = shap.maskers.Image("blur(128,128)", instance[0].shape)

    # By default the Partition explainer is used for all  partition explainer
    explainer = shap.Explainer(model.predict, masker, output_names=class_names)
    # show the values contributing and detracting from each class prediction in order of their prediction probability
    shap_values = explainer(np.expand_dims(instance[0], 0), max_evals=4096, batch_size=64,
                            outputs=shap.Explanation.argsort.flip)
    shap.image_plot(shap_values)


def min_max_normalize_image(image):
    image = tf.reduce_sum(image, -1)

    image -= tf.reduce_min(image)
    image /= tf.reduce_max(image)

    return image


def explain_image_classifier_with_saliency(model, instance):
    # instance = np.array(instance).astype(np.double)
    # instance = np.expand_dims(instance, 0)
    instance = tf.convert_to_tensor(instance)
    instance = tf.cast(instance, tf.double)

    result = model.predict(instance)
    max_idx = tf.argmax(result, 1)

    with tf.GradientTape() as tape:
        tape.watch(instance)
        result = model(instance)
        max_score = result[0, max_idx[0]]

    grads = tape.gradient(max_score, instance)

    pos = tf.nn.relu(grads[0])
    neg = tf.nn.relu(tf.zeros_like(grads[0]) - grads[0])

    pos = min_max_normalize_image(pos)
    neg = min_max_normalize_image(neg)

    plt.imshow(pos, 'ocean')
    plt.show()
    plt.imshow(neg, 'ocean')
    plt.show()

    return pos, neg


def labeled_multi_image(rows, n_cols, row_labs=None, col_labs=None, colors=None, class_names=None):
    """
    creates the multi-image ImageGrid with row and column labels.

    rows: a list of tensors with matching dimension 2, the list of rows to display on the grid
    n_cols: number of columns
    row_labs: a list of row labels
    col_labs: column labels
    class_names: a list containing the class names
    """
    colors = [tuple([c / 255 for c in color]) for color in colors] if colors is not None else colors
    from mpl_toolkits.axes_grid1 import ImageGrid
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(n_cols * 2, 2 * len(rows)))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(len(rows), n_cols),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    col_width = (rows[0].shape[1] // n_cols)
    rows = [[row[:, j * col_width:(j + 1) * col_width, :] for j in range(n_cols)] for i, row in enumerate(rows)]

    rows = [im for sublist in rows for im in sublist]
    print(grid.get_geometry())
    for i, (ax, im) in enumerate(zip(grid, rows)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(col_labs[i % n_cols])
        ax.set_ylabel(row_labs[i // n_cols])
    fig.legend(handles=[mpatches.Patch(color=color, label=class_name)
                        for color, class_name in zip(colors, class_names)])
    plt.show()


def to_shape(a, shape):
    if len(shape) == 3:
        y_, x_, _ = shape
        y, x, _ = a.shape
    else:
        _, y_, x_, _ = shape
        _, y, x, _ = a.shape
    y_pad = (y_ - y)
    x_pad = (x_ - x)

    a, _ = tf.linalg.normalize(a, 1, axis=-1)

    if len(shape) == 3:
        return np.pad(a, (
            (y_pad // 2, y_pad // 2 + y_pad % 2),
            (x_pad // 2, x_pad // 2 + x_pad % 2),
            (0, 0)
        ),
                      mode='constant')
    return np.pad(a, (
        (0, 0),
        (y_pad // 2, y_pad // 2 + y_pad % 2),
        (x_pad // 2, x_pad // 2 + x_pad % 2),
        (0, 0)
    ),
                  mode='constant')


def get_mask(model, image):
    def CE(y_true, y_pred):
        return tf.math.negative(tf.reduce_sum(tf.math.log(y_pred) * y_true, axis=-1))

    def NCE(y_true, y_pred):
        return tf.math.negative(tf.math.log(tf.reduce_mean(y_pred, axis=-1)) - tf.math.log(tf.reduce_max(y_pred, axis=-1)))

    def mask_loss(y_true, y_pred):
        return tf.math.negative(tf.math.log(1 + 2**(-32) - tf.reduce_sum(y_pred, keepdims=True, axis=-1)))

    model = model.get_model()
    # model = tf.keras.models.load_model('../results/stupid_models/1669729179946299', custom_objects={'CE': CE, 'NCE': NCE, 'mask_loss': mask_loss})

    new_outputs = []
    d = -1
    # tf.keras.utils.plot_model(model, '../results/modelpics/' + str(time()).split('.')[-1] + '.png')
    for i, layer in enumerate(model.layers[::-1]):
        if 'cam' in layer.name:
            d = -i - 1
            new_outputs.append(layer.output)
            break
        if 'clam' in layer.name:
            break
    else:
        raise ValueError('cam layer not found')

    for i, layer in enumerate(model.layers):
        try:
            if 'chkpt' in layer.name or 'model' in layer.name:
                new_outputs.append(model.layers[d](layer.output))
            if 'clam' in layer.name:
                new_outputs.append(layer)
        except Exception as e:
            print(e)

    if 'clam' in new_outputs[0].name:
        pred = []
        for model in new_outputs:
            # tf.keras.utils.plot_model(model, '../results/modelpics/' + str(time()).split('.')[-1] + '.png')
            # print(model.layers)
            model.compile()
            p, idk, cam = model.predict(image)
            # print(p)
            p = tf.one_hot(tf.argmax(p, axis=-1), depth=p.shape[-1] + 1)
            p = np.array(p)
            p[:, -1] += 1
            cam = np.array(cam)
            # cam *= p
            # print(p)
            pred.append(cam)

        return np.concatenate([to_shape(z, max([p.shape for p in pred], key=lambda k: k[1])) for z in pred], 2), \
               [tf.reduce_sum(img[0], axis=(0, 1)) for img in pred]
    else:

        model = tf.keras.models.Model(inputs=[model.input], outputs=new_outputs)
        model.compile()
        pred = model.predict(image)
        return np.concatenate([to_shape(z, max([p.shape for p in pred], key=lambda k: k[1])) for z in pred], 2), \
               [tf.reduce_sum(img[0], axis=(0, 1)) for img in pred]


def color_squish(x):
    """
    squishes a tensor of (rows, columns, channels) to a tensor of (rows, columns, RGB (3)) using the color alphabet
    channels must be less than or equal to 26.
    """
    # a couple candidate colors
    colors = [(240, 163, 255), (0, 117, 220), (153, 63, 0), (76, 0, 92), (25, 25, 25), (0, 92, 49),
              (43, 206, 72), (255, 204, 153), (128, 128, 128), (148, 255, 181), (143, 124, 0), (157, 204, 0),
              (194, 0, 136), (0, 51, 128), (255, 164, 5), (255, 168, 187), (66, 102, 0), (255, 0, 16),
              (94, 241, 242), (0, 153, 143), (224, 255, 102), (116, 10, 255), (153, 0, 0), (255, 255, 128),
              (255, 255, 0), (255, 80, 5)][:x.shape[-1]]
    colors = np.array(colors, np.float32)
    x = tf.cast(x, tf.float32)

    return np.array(tf.einsum('ijk,kl->ijl', x, colors)).astype(np.uint8), colors


def show_mask(dset, num_images, model, class_names, fname=''):
    """
    show the composite mask, image, masked image figure for thrifty cam networks

    :param dset: dataset from which to draw examples to explain
    :param num_images: number of images to draw
    :param model: modeldata for the model to explain
    :param class_names: list of class names (in order of their integer value)
    :param fname:
    """
    from PIL import Image
    values = []
    masks = []
    none = []
    imgs = []

    i = 0
    for x, y in iter(dset):
        if isinstance(num_images, int):
            imgs.append(x)
            output, probs = get_mask(model, x)
            # print(output.shape)
            if len(output.shape) < 4:
                output = np.expand_dims(output, 0)
                probs = np.expand_dims(probs, 0)
            values.append(probs)
            masks.append(output[:, :, :, :-1])
            none.append(output[:, :, :, -1])
            num_images -= 1
            if num_images <= 0:
                break
        if isinstance(num_images, list):
            if i in num_images:
                imgs.append(x)
                output, probs = get_mask(model, x)
                # print(output.shape)
                if len(output.shape) < 4:
                    output = np.expand_dims(output, 0)
                    probs = np.expand_dims(probs, 0)
                values.append(probs)
                masks.append(output[:, :, :, :-1])
                none.append(output[:, :, :, -1])
            i += 1

    # print([mask.shape for mask in masks])

    for i, img in enumerate(none):
        ima = imgs[i]
        ima = ima[0] + np.max(np.min(ima[0]), 0)
        ima = ima - np.max(np.min(ima), 0)
        ima = (ima / np.max(ima)) * 255
        ima = np.concatenate([ima for i in range(len(values[i]))], 1)
        # print([mask.shape for mask in masks])
        im = masks[i][0]

        img = np.float32(img[0])
        # print(im.shape)
        img = np.stack([img for i in range(im.shape[-1])], -1)

        im = tf.nn.relu(im - (img / len(class_names)))
        im, colors = color_squish(im)
        # print(im.shape, ima.shape)
        img = (tf.cast(im, tf.float32) + ima) * .5

        img = tf.cast(img, tf.uint8)
        ima = tf.cast(ima, tf.uint8)

        label = values[i]
        normed = [[round(float(ele), 3) for ele in tf.linalg.normalize(lab[:-1], 1)[0]] + ['n/a'] for lab in label]

        def to_label(counts, normed):
            counts = [[l_2 for l_2 in str(np.array(l, int)).replace(']', '').replace('[', '').split(' ') if l_2]
                      for l in counts]

            normed = [[l_2.strip("'") for l_2 in str(np.array(l)).replace(']', '').replace('[', '').split(' ') if l_2]
                      for l in normed]

            labels = ['\n'.join([f'{cl}: {count} p={prob}' for count, prob, cl in zip(c, p, class_names + ['none'])])
                      for c, p in zip(counts, normed)]

            return labels

        labeled_multi_image([img, im, ima], len(to_label(label, normed)), row_labs=['overlay', 'mask', 'image'],
                            col_labs=to_label(label, normed),
                            colors=colors, class_names=class_names)
