import numpy as np
import cv2

class AdaptiveThreshold:
    """
    Applies adaptive thresholding to an image using either mean or Gaussian methods.
    Works with a single-channel grayscale image (2D NumPy array).
    """

    def __init__(self, image_array):
        """
        Initializes the adaptive thresholding class.

        Parameters
        ----------
        image_array : np.ndarray
            Input image array (should be 2D grayscale). If 3-channel, it will convert to grayscale.
        """
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Convert RGB/BGR to grayscale
            self.img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        elif len(image_array.shape) == 2:
            self.img = image_array
        else:
            raise ValueError("Input image must be either a 2D grayscale or 3-channel RGB/BGR image.")

    def mean_threshold(self, block_size=11, c=2):
        """
        Performs adaptive thresholding using the mean of the local neighborhood.

        Parameters
        ----------
        block_size : int, optional
            Size of the neighborhood for computing the mean (must be odd).
        c : int or float, optional
            Constant subtracted from the mean to determine threshold.

        Returns
        -------
        np.ndarray
            Binary thresholded image (uint8, 0 or 255).
        """
        assert block_size % 2 == 1 and block_size > 0, "block_size must be an odd positive integer"
        height, width = self.img.shape
        binary = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                x_min = max(0, i - block_size // 2)
                y_min = max(0, j - block_size // 2)
                x_max = min(height - 1, i + block_size // 2)
                y_max = min(width - 1, j + block_size // 2)
                block = self.img[x_min:x_max+1, y_min:y_max+1]
                thresh = np.mean(block) - c
                if self.img[i, j] >= thresh:
                    binary[i, j] = 255

        return binary

    def gaussian_threshold(self, block_size=11, c=2):
        """
        Performs adaptive thresholding using a Gaussian-weighted sum of the local neighborhood.

        Parameters
        ----------
        block_size : int, optional
            Size of the neighborhood for computing the Gaussian blur (must be odd).
        c : int or float, optional
            Constant subtracted from the Gaussian result to determine threshold.

        Returns
        -------
        np.ndarray
            Binary thresholded image (uint8, 0 or 255).
        """
        assert block_size % 2 == 1 and block_size > 0, "block_size must be an odd positive integer"

        threshold_matrix = cv2.GaussianBlur(self.img, (block_size, block_size), 0)
        threshold_matrix = threshold_matrix - c

        binary = np.zeros_like(self.img, dtype=np.uint8)
        binary[self.img >= threshold_matrix] = 255
        return binary

    def show(self, binary_img, title="Adaptive Threshold"):
        """
        Displays a binary image using OpenCV.

        Parameters
        ----------
        binary_img : np.ndarray
            Binary thresholded image to display.
        title : str, optional
            Window title.
        """
        cv2.imshow(title, binary_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
