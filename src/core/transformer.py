"""Module for image pre-processing."""
import cv2
import os, glob, shutil


class ImageTransformer(object):
    """Transformer class to pre-process images.

    Args:
    ----
    image_path (str): Path for directory where train, test and validation directories are located.
    """

    def __init__(self, image_path):
        self.image_path = image_path
        self.transformed_data_path = "transformed_data"

        if os.path.isdir(self.transformed_data_path) is False:
            for dataset in "train", "test", "valid":
                shutil.copytree(
                    str(os.path.join(self.image_path, f"{dataset}/images")),
                    f"transformed_data/{dataset}",
                )
        else:
            print(f"'{self.transformed_data_path}' directory already exists.")

    def transform_grayscale(self):
        """Remove colours from RGB image and save to directory."""
        for dataset in "train", "test", "valid":
            print(f"Transform {dataset} data.")
            for fil in glob.glob(f"{self.transformed_data_path}/{dataset}/*.jpg"):
                try:
                    image = cv2.imread(fil)
                    gray_image = cv2.cvtColor(
                        image, cv2.COLOR_BGR2GRAY
                    )  # convert to greyscale
                    cv2.imwrite(fil, gray_image)
                except:
                    print("{} is not converted to gray scale")

    def transform_standardise(self):
        pass

    def transform_enhancement(self):
        pass

    def transform_remove_noise(self, is_gray=True, noise_removal_intensity=20):
        """Remove noise from image.

        Args
        ----
        is_gray (bool): Flag if the target image is in grayscale or RGB
        noise_removal_intensity (int): Strenght of noise removal. High integer results in more noise removal however can also remove details of the image.
        """
        for dataset in "train", "test", "valid":
            print(f"Transform {dataset} data.")
            for fil in glob.glob(f"{self.transformed_data_path}/{dataset}/*.jpg"):
                try:
                    image = cv2.imread(fil)
                    if is_gray:
                        denoised_image = cv2.fastNlMeansDenoising(
                            image,
                            h=noise_removal_intensity,
                            templateWindowSize=7,
                            searchWindowSize=21,
                        )  # convert greyscale image
                    else:
                        denoised_image = cv2.fastNlMeansDenoisingColored(
                            image, None, 10, 10, 7, 21
                        )  # convert greyscale image
                    cv2.imwrite(fil, denoised_image)
                except:
                    print("{} is not removed from noise")

    def transform_edge_enhancement(self):
        pass

    def transform_gaussian_blurring(self):
        pass

        def __repr__(self):
            return "Image transformer"
