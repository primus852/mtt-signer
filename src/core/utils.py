"""Helper module."""
import tensorflow as tf

from timeit import default_timer as timer


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)


# TODO: Add functionality of adding multiple metrics
class CollectBatchStats(tf.keras.callbacks.Callback):
    """Class for collecting statistics about training metrics."""

    def __init__(self):
        super().__init__()
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs["loss"])
        self.batch_acc.append(logs["acc"])
        self.model.reset_metrics()
