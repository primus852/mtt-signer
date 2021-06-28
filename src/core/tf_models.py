"""Module to specify tensorflow models."""
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras


# Model paths
# "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
# "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"

# Extractor
# "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"


from timeit import default_timer as timer


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)


class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    # Loss and Error Printing Callback

    def on_train_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_test_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )


class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


class TF_BaseModel(object):
    """Tensorflow hub model class."""

    # TODO: Check options for argument dict
    # TODO: Check how to use variable channels for same model (mobilenet needs 3 but also possible with grayscale?)

    def __init__(self, model_link, image_shape, trainable, n_target):
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
        # self.model =tf.keras.Sequential()
        # hublayer = hub.KerasLayer(self.model_link, input_shape =image_shape, trainable = self.trainable)
        # self.model.add(hublayer)
        # self.model.add(tf.keras.layers.Dense(128, activation='softmax'))

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=image_shape),
                hub.KerasLayer(self.model_link, trainable=self.trainable),
                tf.keras.layers.Flatten(),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(
                    300, activation="elu", kernel_initializer="he_normal"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(
                    100, activation="elu", kernel_initializer="he_normal"
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(
                    self.n_target,
                    activation="softmax",
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                ),
            ]
        )

        # self.model = tf.keras.Sequential(
        # [
        # hub.KerasLayer(
        # self.model_link,
        # input_shape=image_shape,
        # trainable=self.trainable,
        # arguments=dict(batch_norm_momentum=0.997)
        # ),
        # ADD new layers here e.g. batch_normalization, drop_out
        # Layernormalization (https://colab.research.google.com/github/tensorflow/addons/blob/master/docs/tutorials/layers_normalizations.ipynb#scrollTo=Fh-Pp_e5UB54)
        # tf.keras.layers.LayerNormalization(axis=3 , center=True , scale=True),
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dropout(0.2),

        # Batch Norminalization
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Dense(self.n_target, activation='softmax',kernel_initializer="glorot_uniform")
        # ],
        # )
        self.model.build(image_shape)
        self.model.summary()
        # self.history = None

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
        callbacks: list,
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
                for x, y in train_dataset
            ]
        )
        y_train = np.array([y for x, y in train_dataset])
        x_val = np.array(
            [
                tf.image.resize(x, [self.image_shape[0], self.image_shape[1]])
                for x, y in val_dataset
            ]
        )
        y_val = np.array([y for x, y in val_dataset])

        # TODO: Add validation_split as parameter instead of validation dataset
        # TODO: Add callback and earlystopping
        self.history = self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            # verbose set to 2 for avoiding slow callbacks
            verbose=2,
            callbacks=callbacks,
        )

    def predict(self, prediction_dataset):
        """Predict classes."""
        predicted_batch = self.model.predict(prediction_dataset)
        return np.argmax(predicted_batch, axis=-1)

    def evaluate(self):
        pass

    def __repr__(self):
        return "Tensorflow Hub Model Base class"
