import matplotlib.pyplot as plt
import numpy as np

import os

from lime import lime_image
import shap

from skimage.io import imread
from skimage.segmentation import mark_boundaries

import tensorflow as tf

def get_rf(model, instance, patch_size=8, stride=3, randrange=(-1, 1), layer=-4):
    new_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[layer].output[0])
    
    # Create function to apply a random patch on an image tiled across the whole image to get the receptive field of a  
    bottom, top = randrange
    def apply_random_patch(image, top_left_x, top_left_y, patch_size):
        patched_image = np.array(image, copy=True)
        patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, :] = tf.random.uniform((patch_size, patch_size, image.shape[-1]), maxval=top, minval=bottom)

        return patched_image
    
    img = instance[0]
    
    sensitivity_map = np.zeros((img.shape[0], img.shape[1], new_model.output.shape[-1]), dtype=float)
    original_features = new_model(instance)

    # Iterate the patch over the image
    for top_left_x in range(0, img.shape[0] - patch_size, stride):
        for top_left_y in range(0, img.shape[1] - patch_size, stride):
            patched_image = apply_random_patch(img, top_left_x, top_left_y, patch_size)
            predicted_classes = new_model.predict(np.array([patched_image]), verbose=0)
            # Save difference for this specific patched image in map
            sensitivity_map[
                top_left_y:top_left_y + patch_size,
                top_left_x:top_left_x + patch_size,
            ] = tf.reduce_sum(tf.abs(tf.subtract(original_features, predicted_classes)), axis=(1, 2))  # L1 difference
            
    return sensitivity_map


def special_explain_image_classifier_with_lime(model, instance, n_classes):
    """
    show a visual explanation using LIME for an image classification keras Model and a image instance with matplotlib
    """
    instance = np.array(instance)
    explainer = lime_image.LimeImageExplainer(kernel_width=.125)
    
    def special_predict(instance):
        return model.predict(instance)['crossentropy']
    
    explanation = explainer.explain_instance(instance.astype(np.double), special_predict, top_labels=n_classes,
                                             hide_color=0, num_samples=2048, batch_size=16)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=True)
    plt.imshow(mark_boundaries(temp / 2 + .5, mask))
    plt.show()


def special_explain_image_classifier_with_shap(model, instance, class_names):
    instance = np.array(instance).astype(np.double)
    print(instance.shape)
    # define a masker that is used to mask out partitions of the input image, this one uses a blurred background
    masker = shap.maskers.Image("blur(128,128)", instance[0].shape)
    
    def special_predict(instance):
        return model.predict(instance)['crossentropy']
    
    # By default the Partition explainer is used for all  partition explainer
    explainer = shap.Explainer(special_predict, masker, output_names=class_names)
    # show the values contributing and detracting from each class prediction in order of their prediction probability
    shap_values = explainer(np.expand_dims(instance[0], 0), max_evals=4096, batch_size=64,
                            outputs=shap.Explanation.argsort.flip)
    shap.image_plot(shap_values)
    

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
    # image = 

    image -= tf.reduce_min(image)
    image /= tf.reduce_max(image)

    return image



def explain_image_classifier_with_saliency(model, instance):
    # instance = np.array(instance).astype(np.double)
    # instance = np.expand_dims(instance, 0)
    instance = tf.convert_to_tensor(instance)
    instance = tf.cast(instance, tf.double)

    result = model.predict(instance)
    max_idx = tf.argmax(result['crossentropy'], 1)

    with tf.GradientTape() as tape:
        tape.watch(instance)
        result = model(instance)
        print(result, max_idx)
        max_score = result['crossentropy'][0, max_idx[0]]

    grads = tape.gradient(max_score, instance)

    pos = tf.nn.relu(grads[0])
    neg = tf.nn.relu(tf.zeros_like(grads[0]) - grads[0])

    pos = min_max_normalize_image(tf.reduce_max(pos, -1))
    neg = min_max_normalize_image(tf.reduce_max(neg, -1))

    plt.imshow(pos, 'viridis')
    plt.show()
    plt.imshow(neg, 'viridis')
    plt.show()
    print(pos.shape, instance[0].shape)
    plt.imshow(min_max_normalize_image(tf.expand_dims(pos, -1) + instance[0]))
    plt.show()

    return pos, neg


def explain_image_classifier_with_occlusion(model, instance, patch_size=8):
    # code adapted from https://gist.github.com/RaphaelMeudec/7985b0c5eb720a29021d52b0a0be549a
    # Create function to apply a grey patch on an image
    def apply_grey_patch(image, top_left_x, top_left_y, patch_size):
        patched_image = np.array(image, copy=True)
        patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, :] = tf.reduce_mean(image)

        return patched_image
    
    img = instance[0]

    sensitivity_map = np.zeros((img.shape[0], img.shape[1]), dtype=float)

    # Iterate the patch over the image
    for top_left_x in range(0, img.shape[0], patch_size // 2):
        for top_left_y in range(0, img.shape[1], patch_size // 2):
            patched_image = apply_grey_patch(img, top_left_x, top_left_y, patch_size)
            predicted_classes = model.predict(np.array([patched_image]), verbose=0)['crossentropy'][0]
            confidence = predicted_classes[tf.argmax(predicted_classes)]

            # Save confidence for this specific patched image in map
            sensitivity_map[
                top_left_y:top_left_y + patch_size,
                top_left_x:top_left_x + patch_size,
            ] = confidence
            
    plt.imshow(min_max_normalize_image(sensitivity_map), 'viridis')
    plt.show()
    plt.imshow(min_max_normalize_image(tf.cast(tf.expand_dims(min_max_normalize_image(sensitivity_map), -1), tf.float32) + tf.cast(img, tf.float32)))
    plt.show()


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
    model = model.get_model()
    new_outputs = []
    d = -1

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
            model.compile()
            p, idk, cam = model.predict(image)
            p = tf.one_hot(tf.argmax(p, axis=-1), depth=p.shape[-1] + 1)
            p = np.array(p)
            p[:, -1] += 1
            cam = np.array(cam)
            # cam *= p
            pred.append(cam)

        return np.concatenate([to_shape(z, max([p.shape for p in pred], key=lambda k: k[1])) for z in pred], 2), \
               [tf.reduce_sum(img, axis=(1, 2)) for img in pred]
    else:

        model = tf.keras.models.Model(inputs=[model.input], outputs=new_outputs)
        model.compile()
        pred = model.predict(image)
        return np.concatenate([to_shape(z, max([p.shape for p in pred], key=lambda k: k[1])) for z in pred], 2), \
               [tf.reduce_sum(img, axis=(1, 2)) for img in pred]


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

    for x, y in iter(dset):
        imgs.append(x)
        output, probs = get_mask(model, x)
        probs = probs[0]
        if len(output.shape) < 4:
            output = np.expand_dims(output, 0)
            probs = np.expand_dims(probs, 0)

        masks.append(output[:, :, :, :-1])
        none.append(output[:, :, :, -1])
        num_images -= 1
        if num_images <= 0:
            break

    # print([mask.shape for mask in masks])

    for i, img in enumerate(none):
        ima = imgs[i]
        ima = ima[0] + np.max(np.min(ima[0]), 0)
        ima = ima - np.max(np.min(ima), 0)
        ima = (ima / np.max(ima)) * 255
        ima = np.concatenate([ima for i in range(len(values[i]))], 1)
        im = masks[i][0]

        img = np.float32(img[0])
        img = np.stack([img for i in range(im.shape[-1])], -1)

        im = tf.nn.relu(im - (img / len(class_names)))
        im, colors = color_squish(im)
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

        labeled_multi_image([img, im, ima], len(to_label(label, normed)), row_labs=['overlay', 'mask', 'input image'],
                            col_labs=to_label(label, normed),
                            colors=colors, class_names=class_names)
        

def masking_evaluation(model, dset, class_names, n=1):
    """
    return the best and worst cases for masking
    
    I want:
    - orthogonal cases (cosine similarity change)
    - cases where the predicted probability is reduced the most
    - colinear cases
    - cases in which the predicted probability is reduced the least
    """
    
    values = []
    masks = []
    none = []
    imgs = []
    worst_acc, best_acc = [], []
    values_worst, values_best = [], []
    masked_all_worst, masked_all_best = [], []
    
    keras_model = model.get_model()
    
    for cl_idx, cl in enumerate(class_names):
        remaining = n
        print(cl)
        for x, y in iter(dset):
            # model data operations
            pred = keras_model.predict(x)

            prob, cosine, masked_all = pred['crossentropy'], tf.reduce_sum(pred['cosine'], keepdims=True, axis=-1), pred['all_masked']

            output, probs = get_mask(model, x)
            probs = tf.squeeze(probs, 0)
            # collecting only the elements with the correct true class
            there = tf.squeeze(tf.where(prob[:, cl_idx] > .75))
            masked_all = tf.gather(masked_all, there)
            prob = tf.gather(prob, there)
            x = tf.gather(x, there)
            probs = tf.gather(probs, there)
            output = tf.gather(output, there)

            # if there are too few then pick another batch
            if tf.shape(masked_all)[0] < 2:
                print(f'insufficient examples of {cl}')
                continue
            # then collect the indicies from the batches with only the correct class
            probs = tf.squeeze(probs)
            print(tf.shape(prob))
            
            # met = tf.reduce_sum(tf.linalg.normalize(masked_all, axis=-1)[0] * tf.linalg.normalize(prob, axis=-1)[0], keepdims=True, axis=-1)
            met = prob[:, cl_idx] - masked_all[:, cl_idx]
            
            if tf.reduce_max(met) < .1:
                continue
            
            ind = tf.argmax(met)
            best_acc.append(tf.squeeze(ind))
            values_best.append(tf.expand_dims(probs[tf.cast(tf.squeeze(ind), tf.int32)], 0))
            masked_all_best.append(tf.expand_dims(masked_all[tf.cast(tf.squeeze(ind), tf.int32)], 0))

            ind = tf.argmin(met)
            worst_acc.append(tf.squeeze(ind))
            values_worst.append(tf.expand_dims(probs[tf.cast(tf.squeeze(ind), tf.int32)], 0))
            masked_all_worst.append(tf.expand_dims(masked_all[tf.cast(tf.squeeze(ind), tf.int32)], 0))

            if len(output.shape) < 4:
                output = np.expand_dims(output, 0)
                probs = np.expand_dims(probs, 0)

            masks.append(output[:, :, :, :-1])
            none.append(output[:, :, :, -1])
            imgs.append(x)

            remaining -= 1

            if not remaining:
                break
    
    print('worst_acc')
    inds = worst_acc
    values = values_worst
    masked_all = masked_all_best
    for i, img in enumerate(none):
        ima = imgs[i]
        ima = ima[inds[i]] + np.max(np.min(ima[inds[i]]), 0)
        ima = ima - np.max(np.min(ima), 0)
        ima = (ima / np.max(ima)) * 255
        # ima = np.concatenate([ima for i in range(len(values[i]))], 1)
        im = masks[i][inds[i]]

        img = np.float32(img[inds[i]])
        img = np.stack([img for i in range(im.shape[-1])], -1)

        im = tf.nn.relu(im - (img / len(class_names)))
        im, colors = color_squish(im)

        img = (tf.cast(im, tf.float32) + ima) * .5

        img = tf.cast(img, tf.uint8)
        ima = tf.cast(ima, tf.uint8)

        label = values[i]
        
        print(masked_all[i])
        normed = [[round(float(ele), 3) for ele in tf.linalg.normalize(lab[:-1], 1)[0]] + ['n/a'] for lab in label]

        def to_label(counts, normed):
            counts = [[l_2 for l_2 in str(np.array(l, int)).replace(']', '').replace('[', '').split(' ') if l_2]
                      for l in counts]

            normed = [[l_2.strip("'") for l_2 in str(np.array(l)).replace(']', '').replace('[', '').split(' ') if l_2]
                      for l in normed]

            labels = ['\n'.join([f'{cl}: {count} p={prob}' for count, prob, cl in zip(c, p, class_names + ['none'])])
                      for c, p in zip(counts, normed)]

            return labels

        labeled_multi_image([img, im, ima], len(to_label(label, normed)), row_labs=['overlay', 'mask', 'input image'],
                            col_labs=to_label(label, normed),
                            colors=colors, class_names=class_names)
    print('best_acc')
    inds = best_acc
    values = values_best
    masked_all = masked_all_best
    for i, img in enumerate(none):
        ima = imgs[i]
        ima = ima[inds[i]] + np.max(np.min(ima[inds[i]]), 0)
        ima = ima - np.max(np.min(ima), 0)
        ima = (ima / np.max(ima)) * 255
        ima = np.concatenate([ima for i in range(len(values[i]))], 1)
        im = masks[i][inds[i]]

        img = np.float32(img[inds[i]])
        img = np.stack([img for i in range(im.shape[-1])], -1)

        im = tf.nn.relu(im - (img / len(class_names)))
        im, colors = color_squish(im)
        img = (tf.cast(im, tf.float32) + ima) * .5

        img = tf.cast(img, tf.uint8)
        ima = tf.cast(ima, tf.uint8)

        label = values[i]

        print(masked_all[i])
        normed = [[round(float(ele), 3) for ele in tf.linalg.normalize(lab[:-1], 1)[0]] + ['n/a'] for lab in label]

        def to_label(counts, normed):
            counts = [[l_2 for l_2 in str(np.array(l, int)).replace(']', '').replace('[', '').split(' ') if l_2]
                      for l in counts]

            normed = [[l_2.strip("'") for l_2 in str(np.array(l)).replace(']', '').replace('[', '').split(' ') if l_2]
                      for l in normed]

            labels = ['\n'.join([f'{cl}: {count} p={prob}' for count, prob, cl in zip(c, p, class_names + ['none'])])
                      for c, p in zip(counts, normed)]

            return labels

        labeled_multi_image([img, im, ima], len(to_label(label, normed)), row_labs=['overlay', 'mask', 'input image'],
                            col_labs=to_label(label, normed),
                            colors=colors, class_names=class_names)
        