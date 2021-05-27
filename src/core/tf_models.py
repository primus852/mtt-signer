"""Module to specify tensorflow models."""
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# Model paths
# "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
# "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"


class TF_BaseModel(object):
    """Tensorflow hub model class."""

    # TODO: Check if multiple output_keys ['detection_class_entities',
    # 'detection_class_names', 'detection_boxes', 'detection_scores', 'detection_class_labels'] can be specified
    def __init__(self, model_link, image_shape, channels):
        "Initialise class."
        self.model_link = model_link
        self.channels = channels
        self.image_shape = image_shape
        self.model = tf.keras.Sequential(
            [hub.KerasLayer(self.model_link, output_key="detection_scores")]
        )
        self.history = None

    # TODO: Test optimisation on different optimizers, losses and metrics
    def train(
        self,
        train_dataset,
        val_dataset,
        epochs: int,
        batch_size: int,
        optimizer: str,
        loss: str,
        metrics: str,
    ):
        """
        Train model.

        Args:
        ----
        train_dataset (tf.Dataset): Training dataset containing x and y data.
        val_dataset (tf.Dataset): Validation dataset containing x and y data.
        epochs (int): Number of epochs for training
        batch_size (int): Number of instances in each batch.
        optimizer (str): Optimizer to be used for backpropagation
        loss (str): Loss metric for optimisation
        metrics (str): Evaluation metric

        """

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[metrics],
        )

        x_train = np.array([x for x, y, z in train_dataset])
        y_train = np.array([y for x, y, z in train_dataset])
        x_val = np.array([x for x, y, z in val_dataset])
        y_val = np.array([y for x, y, z in val_dataset])

        self.history = self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
        )

    def predict(self):
        pass

    def evaluate(self):
        pass

    def __repr__(self):
        return "Tensorflow Hub Model Base class"
