"""Helper module."""
import tensorflow as tf


# TODO: Add functionality of adding multiple metrics
class CollectBatchStats(tf.keras.callbacks.Callback):
    """Class for collecting statistics about training metrics."""

    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs["loss"])
        self.batch_acc.append(logs["acc"])
        self.model.reset_metrics()
