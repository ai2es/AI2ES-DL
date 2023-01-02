from tensorflow.keras.callbacks import Callback
import keras.backend as K


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
