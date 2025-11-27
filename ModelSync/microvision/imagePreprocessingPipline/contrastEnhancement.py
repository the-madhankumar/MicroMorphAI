import numpy as np
import cv2
import matplotlib.pyplot as plt


class CLAHEEnhancer:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to an image. Accepts either a grayscale NumPy array or an image path.
    """

    def __init__(self, image=None, grayscale_mat=None):
        """
        Initializes the CLAHE enhancer.

        Parameters
        ----------
        image : str, optional
            File path to an image.
        
        grayscale_mat : np.ndarray, optional
            A grayscale image matrix (2D array).
        """
        self.image_path = image
        self.grayscale_mat = grayscale_mat

    def _load_image(self):
        """
        Loads and returns an RGB image.

        Returns
        -------
        np.ndarray
            RGB image matrix.
        """
        if self.image_path is not None:
            img = cv2.imread(self.image_path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(self.grayscale_mat.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    def split_lab(self):
        """
        Converts the image to LAB color space and splits its channels.

        Returns
        -------
        tuple
            L, A, B channels from LAB image.
        """
        img = self._load_image()
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        return cv2.split(lab)

    def apply_clahe(self, clip_limit=5.0, tile_grid_size=(12, 12)):
        """
        Applies CLAHE to the L channel of the LAB image.

        Parameters
        ----------
        clip_limit : float, optional
            Threshold for contrast limiting.
        
        tile_grid_size : tuple, optional
            Size of grid for histogram equalization.

        Returns
        -------
        np.ndarray
            LAB image with enhanced L channel.
        """
        l, a, b = self.split_lab()
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_l = clahe.apply(l)
        return cv2.merge((enhanced_l, a, b))

    def convert_to_rgb(self):
        """
        Converts the CLAHE-enhanced LAB image back to RGB.

        Returns
        -------
        np.ndarray
            CLAHE-enhanced RGB image.
        """
        lab_enhanced = self.apply_clahe()
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    def show(self, title="CLAHE Enhanced Image", axis="off"):
        """
        Displays the CLAHE enhanced image using matplotlib.

        Parameters
        ----------
        title : str, optional
            Title of the displayed image.
        
        axis : str, optional
            Axis display style, e.g., "on" or "off".
        """
        enhanced_img = self.convert_to_rgb()

        plt.figure(figsize=(6, 6))
        plt.title(title)
        plt.imshow(enhanced_img)
        plt.axis(axis)
        plt.show()