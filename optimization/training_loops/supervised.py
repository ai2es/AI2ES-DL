"""
Supervised training loops
"""

from time import time
from support.data_structures import ModelData


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
