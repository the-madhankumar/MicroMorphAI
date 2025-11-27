import cv2
import numpy as np

class GaussianBlurEnhancer:
    """
    Applies random Gaussian noise and Gaussian blur to an image.
    Can work with a file path or a preloaded NumPy array.
    Internally processes images in float32 and converts to uint8 for display.
    """

    def __init__(self, image_path=None, image_array=None, randomness=42):
        """
        Initializes the enhancer.

        Parameters
        ----------
        image_path : str, optional
            Path to the input image file.
        image_array : np.ndarray, optional
            Preloaded image array (H, W, 1 or 3 channels).
        randomness : int, optional
            Seed for random noise generation.
        """
        self.image_path = image_path
        self.image_array = image_array
        self.randomness = randomness
        np.random.seed(randomness)
        self.img = self._load_image().astype(np.float32)  
        
        if len(self.img.shape) == 2:
            self.img = np.stack([self.img]*3, axis=-1)

    def _load_image(self):
        """
        Loads the image as a NumPy array.

        Returns
        -------
        np.ndarray
            Image array.
        """
        if self.image_array is not None:
            return self.image_array
        elif self.image_path is not None:
            img = cv2.imread(self.image_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {self.image_path}")
            return img
        else:
            raise ValueError("Either image_path or image_array must be provided.")

    def add_noise(self, mean=0, std=20):
        """
        Adds random Gaussian noise to the image.

        Parameters
        ----------
        mean : int or float, optional
            Mean of the Gaussian noise.
        std : int or float, optional
            Standard deviation of the Gaussian noise.

        Returns
        -------
        np.ndarray
            Image with added noise (float32).
        """
        noise = np.random.normal(mean, std, self.img.shape).astype(np.float32)
        noisy_img = self.img + noise
        return noisy_img

    def apply_blur(self, kernel_size=(15, 15), sigma=0):
        """
        Applies Gaussian blur to the noisy image.

        Parameters
        ----------
        kernel_size : tuple, optional
            Size of the Gaussian kernel.
        sigma : int, optional
            Standard deviation of the Gaussian kernel. If 0, it is calculated automatically.

        Returns
        -------
        np.ndarray
            Blurred image as uint8.
        """
        noisy_img = self.add_noise()
        blurred_img = cv2.GaussianBlur(noisy_img, kernel_size, sigma)
        blurred_img_uint8 = np.clip(blurred_img, 0, 255).astype(np.uint8)
        return blurred_img_uint8

    def show(self, title="Gaussian Blur Enhanced Image"):
        """
        Displays the blurred image using OpenCV.

        Parameters
        ----------
        title : str, optional
            Window title.
        """
        result = self.apply_blur()
        cv2.imshow(title, result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, path="output.jpg"):
        """
        Saves the blurred image to a file.

        Parameters
        ----------
        path : str
            Path to save the image.
        """
        result = self.apply_blur()
        cv2.imwrite(path, result)
