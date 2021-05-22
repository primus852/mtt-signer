"""Create data class."""
import numpy as np
import subprocess

import tensorflow as tf

# Create a dictionary describing the image features.
image_feature_description = {
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
    "image/filename": tf.io.FixedLenFeature([], tf.string),
    "image/format": tf.io.FixedLenFeature([], tf.string),
    "image/height": tf.io.FixedLenFeature([], tf.int64),
    "image/object/bbox/xmax": tf.io.FixedLenFeature([], tf.float32),
    "image/object/bbox/xmin": tf.io.FixedLenFeature([], tf.float32),
    "image/object/bbox/ymax": tf.io.FixedLenFeature([], tf.float32),
    "image/object/bbox/ymin": tf.io.FixedLenFeature([], tf.float32),
    "image/object/class/label": tf.io.FixedLenFeature([], tf.int64),
    "image/object/class/text": tf.io.FixedLenFeature([], tf.string),
    "image/width": tf.io.FixedLenFeature([], tf.int64),
}


# TODO: Do we need the preprocessing more explicitly?
def _parse_image_function(
    example_proto, IMG_SIZE, channels, gray_scale, standardization
):
    # Parse the input tf.train.Example proto using the dictionary above.
    features = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.image.decode_jpeg(features["image/encoded"], channels=3)
    image = tf.cast(image, tf.float32)
    # Resize image
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 3])

    if gray_scale:
        image = tf.image.rgb_to_grayscale(image)

    if standardization:
        image = tf.image.per_image_standardization(image)

    label = tf.cast(features["image/object/class/label"], tf.int32)

    text_label = features["image/object/class/text"]

    # TODO: Add new naming for resulting ds
    return image, label, text_label


def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x


def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(
        x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    )


# TODO: Add more augment functions
class TFDataClass(object):
    """Class for data preparation for TF models."""

    def __init__(self, IMG_SIZE, channels):
        """Initialize class."""
        self.IMG_SIZE = IMG_SIZE
        self.channels = channels

    def download_data(self):
        """Get pre-defined image dataset in TFRecord format from Roboflow."""
        subprocess.run(
            "!curl -L 'https://app.roboflow.com/ds/rNW01yHJ9Y?key=5oX3PzqFLV' > roboflow.zip; unzip roboflow.zip; rm roboflow.zip",
            shell=True,
        )

    def load_data(self, tfrecord_root: str):
        """Load data into Dataset."""
        self.raw_dataset = tf.data.TFRecordDataset([tfrecord_root])

        return self.raw_dataset.map(
            lambda x: _parse_image_function(
                x,
                IMG_SIZE=self.IMG_SIZE,
                channels=self.channels,
                gray_scale=False,
                standardization=False,
            )
        )

    def augment_data(self, dataset, augmentations=[flip], num_parallel_calls=4):
        """Augmentation of dataset. Grayscale and standardisation not included."""
        for f in augmentations:
            # Apply the augmentation, run 4 jobs in parallel.
            augmented_dataset = dataset.map(
                lambda x, y, z: (flip(x), y, z), num_parallel_calls=num_parallel_calls
            )

        return dataset.concatenate(augmented_dataset).shuffle(len(list(dataset)) * 2)

    # TODO: Add validation set
    def split_data(self, dataset, test_size=0.2):
        """Train test split."""
        y_targets = np.array(
            [text.numpy().decode("utf-8") for image, target, text in iter(dataset)]
        )

        unique_labels = np.unique(y_targets)
        dataset_size = len(list(dataset))
        n = 0

        for letter in unique_labels:
            target_dataset = dataset.filter(lambda x, y, z: z == letter)
            target_dataset = target_dataset.shuffle(dataset_size)

            # Split them
            target_test_samples_len = int(len(list(target_dataset)) * test_size)
            target_test = target_dataset.take(target_test_samples_len)
            target_train = target_dataset.skip(target_test_samples_len)

            print(
                f"Train {letter} = ",
                len(list(target_test)),
                " Test {letter} = ",
                len(list(target_train)),
            )

            # Gather datasets
            if n == 0:
                train_dataset = target_train
                test_dataset = target_test
            else:
                train_dataset = train_dataset.concatenate(target_train).shuffle(
                    dataset_size
                )
                test_dataset = test_dataset.concatenate(target_test).shuffle(
                    dataset_size
                )

            n += 1

        return train_dataset, test_dataset
