"""
Self-supervised training loops
"""

import tensorflow as tf
from time import time
from optimization.schedules import diffusion_schedule
from support.data_structures import ModelData


def keras_self_supervised(model,
                          train_dset,
                          val_dset,
                          network_params,
                          experiment_params,
                          callbacks,
                          evaluate_on=None,
                          train_steps=None,
                          val_steps=None
                          ):
    """
    train a keras model on a dataset and evaluate it on other datasets then return the ModelData instance

    :param model: keras model
    :param train_dset: tf.data.Dataset for training
    :param val_dset: tf.data.Dataset for evaluation
    :param network_params: see Config.network_params
    :param experiment_params: see Config.experiment_params
    :param callbacks: Callbacks for keras fit training
    :param evaluate_on: a dictionary of finite objects passable to model.evaluate
    :param train_steps: number of steps per epoch
    :param val_steps: number of validations steps per epoch
    :return: a ModelData instance
    """
    # Override arguments if we are using exp_index

    train_steps = train_steps if train_steps is not None else 100

    print(train_steps, val_steps)

    evaluate_on = dict() if evaluate_on is None else evaluate_on

    start = time()
    print(model.summary())
    # tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True,
    #                          to_file=os.curdir + f'/../visualizations/models/model_{str(time())[:6]}.png')
    # Perform the experiment?
    if experiment_params['nogo']:
        # No!
        print("NO GO")
        return

    # Learn
    #  steps_per_epoch: how many batches from the training set do we use for training in one epoch?
    #  validation_steps=None
    #  means that ALL validation samples will be used (of the selected subset)
    print(val_steps)
    history = model.fit(train_dset,
                        epochs=experiment_params['epochs'],
                        verbose=True,
                        validation_data=val_dset,
                        callbacks=callbacks,
                        steps_per_epoch=train_steps,
                        shuffle=False)

    evaluate_on = dict() if evaluate_on is None else evaluate_on

    evaluations = {k: model.evaluate(evaluate_on[k], steps=val_steps) for k in evaluate_on}
    end = time()
    # populate results data structure
    model_data = ModelData(weights=model.get_weights(),
                           network_params=network_params,
                           network_fn=network_params['network_fn'],
                           evaluations=evaluations,
                           classes=network_params['network_args']['n_classes'],
                           history=history.history,
                           run_time=end - start)
    print('returning model data')

    return model_data


"""
def keras_supervised(model,
                     train_dset,
                     val_dset,
                     network_params,
                     experiment_params,
                     callbacks,
                     evaluate_on=None,
                     train_steps=None,
                     val_steps=None
                     ):
                     
    history = model.fit(train_dset,
                        epochs=experiment_params['epochs'],
                        verbose=True,
                        validation_data=val_dset,
                        callbacks=callbacks,
                        steps_per_epoch=train_steps,
                        shuffle=False)

def diffusion(model,
              train_dset,
              val_dset,
              network_params,
              experiment_params,
              callbacks,
              evaluate_on=None,
              train_steps=None,
              val_steps=None):
    # network parameters: image_size
    # unfilled parameters: batch_size, n_gpu
    normalizer = tf.keras.layers.Normalization()
    network = model
    ema_network = tf.keras.models.clone_model(network)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # TODO: make this suitable outside of keras
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def train_step(model, images):
        # normalize images to have standard deviation of 1, like the noises
        images = normalizer(images, training=True)
        noises = tf.random.normal(shape=(batch_size // n_gpu, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size // n_gpu, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = loss(noises, pred_noises)  # used for training (should always be MSE)
            image_loss = loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, network.trainable_weights)
        optimizer.apply_gradients(zip(gradients, network.trainable_weights))

        # not sure what these will look like at training time
        noise_loss_tracker.update_state(noise_loss)
        image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(network.weights, ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        # this is not super relevant outside of pure keras training
        return {m.name: m.result() for m in self.metrics[:-1]}
"""
