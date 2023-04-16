from tensorflow.keras.callbacks import Callback
import keras.backend as K
import numpy as np
import logging


class LossWeightScheduler(Callback):
    def __init__(self, schedule):
        self.schedule = schedule
        self.alpha = None

    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        if epoch == 2:
            try:
                self.alpha = K.get_value(self.alpha)
            except:
                self.alpha = K.variable(.5)
        self.alpha = self.schedule(epoch)


class EarlyStoppingDifference(Callback):
    """Stop training when the difference between two monitored metrics has stopped improving."""

    def __init__(
        self,
        metric_0="val_loss",
        metric_1="val_categorical_accuracy",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
        n_classes=3,
    ):
        super().__init__()

        self.metric_0 = metric_0
        self.metric_1 = metric_1
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.start_from_epoch = start_from_epoch
        self.n_classes = n_classes

        if mode not in ["auto", "min", "max"]:
            logging.warning(
                "EarlyStopping mode %s is unknown, fallback to auto mode.",
                mode,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if (
                self.metric_0.endswith("acc")
                or self.metric_0.endswith("accuracy")
                or self.metric_0.endswith("auc")
            ):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None or epoch < self.start_from_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous
            # best.
            if self.baseline is None or self._is_improvement(
                current, self.baseline
            ):
                self.wait = 0

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    print(
                        "Restoring model weights from "
                        "the end of the best epoch: "
                        f"{self.best_epoch + 1}."
                    )
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(
                f"Epoch {self.stopped_epoch + 1}: early stopping"
            )

    def get_monitor_value(self, logs):
        logs = logs or {}
        # this is only really relevant for classification
        monitor_value = (logs.get(self.metric_0) - (1 / self.n_classes)) - abs((1 / self.n_classes) - logs.get(self.metric_1))
        if monitor_value is None:
            logging.warning(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)
    
    
class EarlyStoppingOQ(Callback):
    """
    Stop training when the difference between two monitored metrics has stopped improving.

    Stops when metric_0 - metric_1 reaches a maximum (or minimum based on mode).  Default is maximum.
    """

    def __init__(
        self,
        metric_0="val_loss",
        metric_1="val_categorical_accuracy",
        metric_2="val_categorical_accuracy",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
        n_classes=1,
    ):
        """
        :param metric_0: The first metric to watch, this is the metric subtracted from
        :param metric_1: The second metric to watch, this is the metric that is subtracted
        :param min_delta: Minimum amount for difference to change to reset patience
        :param patience: Number of epochs to wait for metric difference to change
        :param verbose: Verbosity level
        :param mode: "min", "max", or "auto" based on whether to minimize, maximize, or automatically choose between the
                    two for the the metric difference.  Default is "auto."
        :param baseline: Baseline value to start the metric difference at
        :param restore_best_weights: Whether or not to restore best weights when patience is met.  Default is False.
        :param start_from_epoch: Start measuring the metric difference from this epoch. Default is 0.
        :param n_classes:  Number of classes for classification problem.  Default is 1 (not a classification problem)
        """
        super().__init__()

        self.metric_0 = metric_0
        self.metric_1 = metric_1
        self.metric_2 = metric_2
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.start_from_epoch = start_from_epoch
        self.n_classes = n_classes

        if mode not in ["auto", "min", "max"]:
            logging.warning(
                "EarlyStopping mode %s is unknown, fallback to auto mode.",
                mode,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if (
                self.metric_0.endswith("acc")
                or self.metric_0.endswith("accuracy")
                or self.metric_0.endswith("auc")
            ):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None or epoch < self.start_from_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous
            # best.
            if self.baseline is None or self._is_improvement(
                current, self.baseline
            ):
                self.wait = 0

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    print(
                        "Restoring model weights from "
                        "the end of the best epoch: "
                        f"{self.best_epoch + 1}."
                    )
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(
                f"Epoch {self.stopped_epoch + 1}: early stopping"
            )

    def get_monitor_value(self, logs):
        logs = logs or {}
        # this is only really relevant for classification
        monitor_value = ((logs.get(self.metric_0) / self.n_classes) - abs((1 / self.n_classes) - logs.get(self.metric_1)) - logs.get(self.metric_2))
        if monitor_value is None:
            logging.warning(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)