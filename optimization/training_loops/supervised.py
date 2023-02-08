"""
Supervised training loops
"""

from time import time
from support.data_structures import ModelData
import tensorflow as tf
import numpy as np
import wandb


def wandb_supervised(model,
                     train_dset,
                     val_dset,
                     network_params,
                     experiment_params,
                     callbacks,
                     evaluate_on=None,
                     train_steps=None,
                     val_steps=None,
                     hardware_params=None,
                     dataset_params=None,
                     optimization_params=None,
                     **kwargs):
    run = wandb.init(project="test-project", entity="ai2es",
                     config={
                         'experiment': experiment_params,
                         'hardware': hardware_params,
                         'dataset': dataset_params,
                         'network': network_params,
                         'optimization': optimization_params
                     })

    optimizer = model.optimizer
    loss_fn = model.loss

    def train_step(x, y, metrics):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        for train_metric in metrics:
            train_metric.update_state(y, logits)

        return loss_value

    def test_step(x, y, metrics):
        val_logits = model(x, training=False)
        loss_value = loss_fn(y, val_logits)

        for val_metric in metrics:
            val_metric.update_state(y, val_logits)

        return loss_value

    def train(train_dataset, val_dataset, train_metrics, val_metrics,
              epochs=10, train_steps=200, val_steps=50, callbacks=None):

        _callbacks = callbacks if callbacks else []
        callbacks = tf.keras.callbacks.CallbackList(
            _callbacks, add_history=True, model=model)

        early_stop_config = {'cb': None, 'stop_early': False}
        # check for an early stopping callback
        for callback in callbacks:
            if isinstance(callback, tf.keras.callbacks.EarlyStopping):
                early_stop_config['stop_early'] = True
                early_stop_config['cb'] = callback

        logs = {}
        callbacks.on_train_begin(logs=logs)
        for epoch in range(epochs):
            callbacks.on_epoch_begin(epoch, logs=logs)
            print("\nStart of epoch %d" % (epoch,))

            train_loss = []
            val_loss = []

            # Iterate over the batches of the dataset
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                callbacks.on_batch_begin((x_batch_train, y_batch_train), logs=logs)
                callbacks.on_train_batch_begin((x_batch_train, y_batch_train), logs=logs)

                loss_value = train_step(x_batch_train, y_batch_train, train_metrics)
                train_loss.append(float(loss_value))

                logs = {'epochs': epoch,
                        'loss': np.mean(train_loss),
                        **{metric.name: float(metric.result().numpy()) for metric in train_metrics}}

                callbacks.on_train_batch_end((x_batch_train, y_batch_train), logs=logs)
                callbacks.on_batch_end((x_batch_train, y_batch_train), logs=logs)
                if step >= train_steps:
                    break

            # Run a validation loop at the end of each epoch
            for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                callbacks.on_batch_begin((x_batch_val, y_batch_val), logs=logs)
                callbacks.on_test_batch_begin((x_batch_val, y_batch_val), logs=logs)

                val_loss_value = test_step(x_batch_val,
                                           y_batch_val,
                                           val_metrics)
                val_loss.append(float(val_loss_value))

                logs = {'epochs': epoch,
                        'val_loss': np.mean(val_loss),
                        **{f'val_{metric.name}': float(metric.result().numpy()) for metric in val_metrics}}

                callbacks.on_test_batch_end((x_batch_val, y_batch_val), logs=logs)
                callbacks.on_batch_end((x_batch_val, y_batch_val), logs=logs)

                if val_steps:
                    if step >= val_steps:
                        break

            logs = {'epochs': epoch,
                    'loss': np.mean(train_loss),
                    **{metric.name: float(metric.result().numpy()) for metric in train_metrics},
                    'val_loss': np.mean(val_loss),
                    **{f'val_{metric.name}': float(metric.result().numpy()) for metric in val_metrics}}
            print(logs)
            callbacks.on_epoch_end(epoch, logs=logs)

            # ⭐: log metrics using wandb.log
            wandb.log({'epochs': epoch,
                       'loss': np.mean(train_loss),
                       **{metric.name: float(metric.result().numpy()) for metric in train_metrics},
                       'val_loss': np.mean(val_loss),
                       **{f'val_{metric.name}': float(metric.result().numpy()) for metric in val_metrics}})

            # check for early stopping
            if early_stop_config['stop_early']:
                if early_stop_config['cb'].wait >= early_stop_config['cb'].patience and epoch > 0:
                    print('stopping early.')
                    break
            # Reset metrics at the end of each epoch
            for train_metric in train_metrics:
                train_metric.reset_states()
            for val_metric in val_metrics:
                val_metric.reset_states()

    train(train_dset,
          val_dset,
          experiment_params['train_metrics'],
          experiment_params['val_metrics'],
          epochs=experiment_params['epochs'],
          train_steps=train_steps,
          val_steps=val_steps,
          callbacks=callbacks)

    run.finish()


def keras_supervised(model,
                     train_dset,
                     val_dset,
                     network_params,
                     experiment_params,
                     callbacks,
                     evaluate_on=None,
                     train_steps=None,
                     val_steps=None,
                     **kwargs
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
