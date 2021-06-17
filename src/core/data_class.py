# Show the CUDA status
import matplotlib.pyplot as plt
import torch
from IPython.display import Image, clear_output

clear_output()
print(
    "Setup complete. Using torch %s %s"
    % (
        torch.__version__,
        torch.cuda.get_device_properties(0) if torch.cuda.is_available() else "CPU",
    )
)


"""Create data class."""
import numpy as np
import subprocess
import tensorflow as tf
from tensorflow.keras import optimizers


# Create a dictionary describing the image features.
image_feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=27),
}


# TODO: Do we need the preprocessing more explicitly?
def _parse_image_function(
    example_proto, IMG_SIZE, channels, gray_scale, standardization
):
    # Parse the input tf.train.Example proto using the dictionary above.
    features = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.image.decode_jpeg(features["image"], channels=3)
    image = tf.cast(image, tf.float32)
    # Resize image
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 3])

    if gray_scale:
        image = tf.image.rgb_to_grayscale(image)

    if standardization:
        image = tf.image.per_image_standardization(image)

    label = tf.cast(features["label"], tf.int32) - 1

    return image, label


def plot_images(dataset, n_images, samples_per_image):
    output = np.zeros((32 * n_images, 32 * samples_per_image, 3))

    row = 0
    for images in dataset.repeat(samples_per_image).batch(n_images):
        output[:, row * 32 : (row + 1) * 32] = np.vstack(images.numpy())
        row += 1

    plt.figure()
    plt.imshow(output)
    plt.show()


# TODO: Check if flip/rotating will "create" new letter
def flipU(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    # x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x


def flipS(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    # x = tf.image.random_flip_up_down(x)

    return x


# rotate in 0, 90, 180, 270 degrees
def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(
        x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    )


# rotate in random degrees
def rotate_random(x: tf.Tensor) -> tf.Tensor:
    if x.shape.__len__() == 4:

        random_angles = tf.random.uniform(
            shape=(tf.shape(x)[0],), minval=-np.pi / 4, maxval=np.pi / 4
        )
    if x.shape.__len__() == 3:
        random_angles = tf.random.uniform(shape=(), minval=-np.pi / 4, maxval=np.pi / 4)

    return tfa.image.rotate(x, random_angles)


# color distortion
def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_saturation(x, lower=0.5, upper=1.5)
    x = tf.image.random_brightness(x, max_delta=32.0 / 255.0)
    x = tf.image.random_contrast(x, lower=0.5, upper=1.5)
    x = tf.image.random_hue(x, max_delta=0.2)
    return x


def pixelated(x: tf.Tensor) -> tf.Tensor:
    """pixelated augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize(
            [img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(32, 32)
        )
        # Return a random crop
        return crops[
            tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)
        ]

    choice = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))


# TODO: Add more augment functions
class TFDataClass(object):
    """Class for data preparation for TF models."""

    def __init__(self, IMG_SIZE):
        """Initialize class."""
        self.IMG_SIZE = (IMG_SIZE[0], IMG_SIZE[1])
        self.channels = IMG_SIZE[2]
        self.raw_dataset = None

    def download_data(self, file_id="1Ov-XShbcza1Kw3j5wp_yFxkK9E38DSZd"):
        """Get pre-defined image dataset in TFRecord format from Roboflow."""
        popen = subprocess.Popen(
            # "curl -L 'https://app.roboflow.com/ds/rNW01yHJ9Y?key=5oX3PzqFLV' > roboflow.zip; unzip roboflow.zip; rm roboflow.zip",
            f"gdown --id {file_id}; unzip ASL_MTT.zip; rm ASL_MTT.zip",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        out, errs = popen.communicate()
        if popen.returncode != 0:
            print(
                "ErrorCode: %d, PopenStdOut: %s,  PopenStdErr: %s"
                % (popen.returncode, out, errs)
            )

    def load_data(
        self,
        tfrecord_root: str,
        gray_scale: bool = False,
        standardization: bool = False,
    ):
        """Load data into Dataset."""
        for subset in ["train", "valid"]:
            if subset == "train":
                path = f"{tfrecord_root}/train/Letters.tfrecords"
            elif subset == "valid":
                path = f"{tfrecord_root}/valid/Letters.tfrecord"
            temp_df = tf.data.TFRecordDataset([path])
            if self.raw_dataset is None:
                self.raw_dataset = temp_df
            else:

                self.raw_dataset = self.raw_dataset.concatenate(temp_df)

        self.raw_dataset_parsed = self.raw_dataset.map(
            lambda x: _parse_image_function(
                x,
                IMG_SIZE=self.IMG_SIZE[0],
                channels=self.channels,
                gray_scale=gray_scale,
                standardization=standardization,
            )
        )

        return self.raw_dataset_parsed

    def load_test_data(
        self,
        tfrecord_root: str,
        gray_scale: bool = False,
        standardization: bool = False,
    ):
        """Load data into Dataset."""
        raw_dataset = tf.data.TFRecordDataset(
            [f"{tfrecord_root}/test/Letters.tfrecords"]
        )

        raw_dataset_parsed = raw_dataset.map(
            lambda x: _parse_image_function(
                x,
                IMG_SIZE=self.IMG_SIZE[0],
                channels=self.channels,
                gray_scale=gray_scale,
                standardization=standardization,
            )
        )

        return raw_dataset_parsed

    def augment_data(self, dataset, augmentations=[flipU], num_parallel_calls=4):
        """Augmentation of dataset. Grayscale and standardisation not included."""
        augmented_dataset = dataset
        for f in augmentations:
            temp_df = dataset.map(
                lambda x, y: (f(x), y), num_parallel_calls=num_parallel_calls
            )
            augmented_dataset = augmented_dataset.concatenate(temp_df)

        return augmented_dataset.shuffle(len(list(dataset)) * 2)

    # TODO: Speed up process, tf functions?
    def split_data(self, dataset, val_size=0.2, only_subset=0.5):
        """Train test split with stratified data."""

        unique_labels = np.arange(0, 26, 1)
        dataset_size = len(list(dataset))
        n = 0

        for letter in unique_labels:
            target_dataset = dataset.filter(lambda x, y: y == letter)
            if only_subset:
                target_dataset = target_dataset.take(
                    int(len(list(dataset)) * only_subset)
                ).shuffle(int(len(list(dataset)) * only_subset))
            else:
                target_dataset = target_dataset.shuffle(dataset_size)

            target_val_samples_len = int(len(list(target_dataset)) * val_size)
            target_val = target_dataset.take(target_val_samples_len)
            target_train = target_dataset.skip(target_val_samples_len)

            print(
                f"Train label {letter} = ",
                len(list(target_train)),
                f" Val label {letter} = ",
                len(list(target_val)),
            )

            if n == 0:
                train_dataset = target_train
                val_dataset = target_val
            else:
                train_dataset = train_dataset.concatenate(target_train).shuffle(
                    dataset_size
                )
                val_dataset = val_dataset.concatenate(target_val).shuffle(dataset_size)

            n += 1

        return train_dataset, val_dataset
