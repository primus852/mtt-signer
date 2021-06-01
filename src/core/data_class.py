"""Create data class."""
import numpy as np
import subprocess

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
        x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32),
        # Rotate 20 degrees
        x=ImageDataGenerator(rotation_range=20)
    )


# added by Mayco 28.05.2021
def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x


def zoom(x: tf.Tensor) -> tf.Tensor:
    """Zoom augmentation

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
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))


## an other alternative
def distort_color(image, thread_id=0, scope=None):
    """Distort the color of the image.
  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for op_scope.
  Returns:
    color-distorted image
  """
    with tf.op_scope([image], scope, 'distort_color'):
        color_ordering = thread_id % 2

    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


# Todo: we need it?
def distort_image(image, height, width, bbox, thread_id=0, scope=None):
    """Distort one image for training a network.
   Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.
  Args:
    image: 3-D float Tensor of image
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    thread_id: integer indicating the preprocessing thread.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
    with tf.op_scope([image, height, width, bbox], scope, 'distort_image'):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # Display the bounding box in the first thread only.
        if not thread_id:
            image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                          bbox)
        tf.image_summary('image_with_bounding_boxes', image_with_box)

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an allowed
        # range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.05, 1.0],
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        if not thread_id:
            image_with_distorted_box = tf.image.draw_bounding_boxes(
                tf.expand_dims(image, 0), distort_bbox)
            tf.image_summary('images_with_distorted_bounding_box',
                             image_with_distorted_box)

        # Crop the image to the specified bounding box.
        distorted_image = tf.slice(image, bbox_begin, bbox_size)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.
        resize_method = thread_id % 4
        distorted_image = tf.image.resize_images(distorted_image, [height, width],
                                                 method=resize_method)
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([height, width, 3])
        if not thread_id:
            tf.image_summary('cropped_resized_image',
                             tf.expand_dims(distorted_image, 0))

        # Randomly flip the image horizontally.
        # distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Randomly distort the colors.
        distorted_image = distort_color(distorted_image, thread_id)

        if not thread_id:
            tf.image_summary('final_distorted_image',
                             tf.expand_dims(distorted_image, 0))
        return distorted_image


# TODO: Add more augment functions
class TFDataClass(object):
    """Class for data preparation for TF models."""

    def __init__(self, IMG_SIZE):
        """Initialize class."""
        self.IMG_SIZE = (IMG_SIZE[0],IMG_SIZE[1])
        self.channels = IMG_SIZE[2]
        self.raw_dataset= None

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

    def load_data(self, tfrecord_root: str,gray_scale:bool = False, standardization: bool = False):
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
                gray_scale=False,
                standardization=False,
            )
        )

        return self.raw_dataset_parsed

    def augment_data(self, dataset, augmentations=[flip], num_parallel_calls=4):
        """Augmentation of dataset. Grayscale and standardisation not included."""
        for f in augmentations:
            augmented_dataset = dataset.map(
                lambda x, y, z: (flip(x), y, z), num_parallel_calls=num_parallel_calls
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
