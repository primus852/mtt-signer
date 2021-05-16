"""Module to specify tensorflow models."""

import tensorflow as tf
import tensorflow_hub as hub


# Model paths
# "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
# "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"


class TF_BaseModel(object):
    """Tensorflow hub model class."""

    def __init__(self, model_link, image_shape, channels):
        self.model_link = model_link
        self.channels = channels
        self.image_shape = image_shape
        self.model = tf.keras.Sequential(
            [hub.KerasLayer(self.model_link, input_shape=self.image_shape)]
        )

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

    def __repr__(self):
        return "Tensorflow Hub Model Base class"
