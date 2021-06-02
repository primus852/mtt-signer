"""Create data class."""
import numpy as np
import subprocess
import tensorflow_addons as tfa
import tensorflow as tf


# TODO: Add background removal

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


# TODO: Add distortion
# TODO: Check if flip/rotating will "create" new letter
def flipU(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    #x = tf.image.random_flip_left_right(x)
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
    #x = tf.image.random_flip_up_down(x)

    return x

#rotate in 0, 90, 180, 270 degrees
def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

#rotate in random degrees
def rotate_random(x: tf.Tensor) -> tf.Tensor:
    if x.shape.__len__() ==4:
            
        random_angles = tf.random.uniform(shape = (tf.shape(x)[0], ), minval = -np
        .pi / 4, maxval = np.pi / 4)
    if x.shape.__len__() ==3:
        random_angles = tf.random.uniform(shape = (), minval = -np
        .pi / 4, maxval = np.pi / 4)

    return tfa.image.rotate(x,random_angles)

# color distortion
def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_saturation(x, lower=0.5, upper=1.5)
    x = tf.image.random_brightness(x, max_delta=32. / 255.)
    x = tf.image.random_contrast(x, lower=0.5, upper=1.5)
    x = tf.image.random_hue(x, max_delta=0.2)
    return x


def pixelated (x: tf.Tensor) -> tf.Tensor:
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
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices =np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

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

    def download_data(self):
        """Get pre-defined image dataset in TFRecord format from Roboflow."""
        popen = subprocess.Popen(
            "curl -L 'https://app.roboflow.com/ds/rNW01yHJ9Y?key=5oX3PzqFLV' > roboflow.zip; unzip roboflow.zip; rm roboflow.zip",
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
        for subset in ["train", "test", "valid"]:
            temp_df = tf.data.TFRecordDataset(
                [f"{tfrecord_root}/{subset}/Letters.tfrecord"]
            )
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

    def augment_data(self, dataset, augmentations=[flip], num_parallel_calls=4):
        """Augmentation of dataset. Grayscale and standardisation not included."""
        for f in augmentations:
            augmented_dataset = dataset.map(
                lambda x, y, z: (f(x), y, z), num_parallel_calls=num_parallel_calls
            )

        return dataset.concatenate(augmented_dataset).shuffle(len(list(dataset)) * 2)

    # TODO: Add validation set
    # TODO: Speed up process, tf functions?
    def split_data(self, dataset, test_size=0.2):
        """Train test split with stratified data."""
        y_targets = np.array(
            [text.numpy().decode("utf-8") for image, target, text in iter(dataset)]
        )

        unique_labels = np.unique(y_targets)
        dataset_size = len(list(dataset))
        n = 0

        for letter in unique_labels:
            target_dataset = dataset.filter(lambda x, y, z: z == letter)
            target_dataset = target_dataset.shuffle(dataset_size)

            target_test_samples_len = int(len(list(target_dataset)) * test_size)
            target_test = target_dataset.take(target_test_samples_len)
            target_train = target_dataset.skip(target_test_samples_len)

            print(
                f"Train {letter} = ",
                len(list(target_train)),
                f" Test {letter} = ",
                len(list(target_test)),
            )

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