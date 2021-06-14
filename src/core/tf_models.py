"""Module to specify tensorflow models."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# Model paths
# "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
# "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"

# Extractor
# "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"


class TF_BaseModel(object):
    """Tensorflow hub model class."""

    # TODO Check options for argument dict
    # TODO Check how to use variable channels for same model (mobilenet needs 3 but also possible with grayscale?)
    # https://stackoverflow.com/questions/48744666/tensorflow-object-detection-api-1-channel-image
    # https://stackoverflow.com/questions/14786179/how-to-convert-a-1-channel-image-into-a-3-channel-with-opencv2
    # https://askubuntu.com/questions/1091493/convert-a-1-channel-image-to-a-3-channel-image
    # TODO Automatically detect image size which is needed (config file?)
    def __init__(
        self,
        model_link,
        image_shape,
        n_target,
        trainable=False,
        classification_head_activation="softmax",
    ):
        """Initialise class.

        Args:
        ----
        model_link (str): Link to TF Hub model
        image_shape (tuple): Tuple with format (width, height, channels)
        trainable (bool): If true Transfer Learning is applied
        n_target (int): Number of unique target values

        """
        self.model_link = model_link
        self.channels = image_shape[2]
        self.image_shape = image_shape
        self.trainable = trainable
        self.n_target = n_target
        self.model = tf.keras.Sequential(
            [
                hub.KerasLayer(
                    self.model_link,
                    input_shape=image_shape,
                    trainable=self.trainable,
                    # TODO arguments=dict(batch_norm_momentum=0.997)
                ),
                tf.keras.layers.Dense(
                    self.n_target, activation=classification_head_activation
                ),
            ],
        )
        self.history = None

    # TODO Test optimisation on different optimizers, losses and metrics
    def train(
        self,
        train_dataset,
        val_dataset,
        epochs: int,
        batch_size: int,
        optimizer: str,
        loss: str,
        metrics: list,
        callbacks: list,
        plot_results: bool = False,
    ):
        """
        Train model.

        Args:
        ----
        train_dataset (tf.Dataset): Training dataset containing x and y data.
        val_dataset (tf.Dataset): Validation dataset containing x and y data.
        epochs (int): Number of epochs for training
        batch_size (int): Number of instances in each batch.
        optimizer (str/function): Optimizer to be used for backpropagation. Does not work for 'adam' yet
        loss (str/function): Loss metric for optimisation
        metrics (str/function): Evaluation metric

        """

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )

        # Resize image to needed input size for models
        x_train = np.array(
            [
                tf.image.resize(x, [self.image_shape[0], self.image_shape[1]])
                for x, y, z in train_dataset
            ]
        )
        y_train = np.array([y for x, y, z in train_dataset])
        x_val = np.array(
            [
                tf.image.resize(x, [self.image_shape[0], self.image_shape[1]])
                for x, y, z in val_dataset
            ]
        )
        y_val = np.array([y for x, y, z in val_dataset])

        # TODO Add validation_split as parameter instead of validation dataset
        # TODO Add callback and earlystopping
        self.history = self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=2,
            callbacks=callbacks,
        )

        if plot_results:
            pd.DataFrame(self.history.history).plot(figsize=(8, 5))
            plt.grid(True)
            plt.gca().set_ylim(0, 1)
            plt.show()

    def predict(self, prediction_dataset, batch_size=1):
        """Predict classes."""
        predicted_batch = self.model.predict(prediction_dataset.batch(batch_size))
        return np.argmax(predicted_batch, axis=-1)

    def __repr__(self):
        return "Tensorflow Hub Model Base class"
