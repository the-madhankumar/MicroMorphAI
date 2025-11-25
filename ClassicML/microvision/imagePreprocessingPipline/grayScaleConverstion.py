from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class PCAGrayscaleConverter:
    """
    A converter that transforms an RGB image into a grayscale image
    using the principal component corresponding to the highest eigenvalue.
    """

    def __init__(self, image_path: str):
        """
        Initializes the converter by loading the image and creating
        the corresponding numpy matrix.

        Parameters
        ----------
        image_path : str
            Path to the RGB image file.
        """
        self.image_path = image_path
        self.rgb_img = Image.open(image_path)
        self.img_mat = np.array(self.rgb_img)

    def _flatten_image(self) -> np.ndarray:
        """
        Flattens the RGB image into a 2D matrix where each row represents
        a pixel and each column represents a channel.

        Returns
        -------
        np.ndarray
            Flattened image matrix with shape (num_pixels, num_channels).
        """
        return np.reshape(
            self.img_mat,
            (self.img_mat.shape[0] * self.img_mat.shape[1], -1),
            order='C'
        )

    def _compute_pca_components(self, vect_mat: np.ndarray):
        """
        Computes the PCA components of the flattened image through eigen decomposition.

        Parameters
        ----------
        vect_mat : np.ndarray
            Flattened image matrix.

        Returns
        -------
        tuple
            A tuple containing (alpha, beta, gamma) which are the normalized
            principal component weights.
        """
        mean_vals = np.mean(vect_mat, axis=0)
        cen_mat = vect_mat - mean_vals
        cov_mat = (cen_mat.T @ cen_mat) / vect_mat.shape[0]

        eig_val, eig_vec = np.linalg.eig(cov_mat)
        max_idx = np.argmax(eig_val)
        principal_vector = eig_vec[:, max_idx]
        alpha, beta, gamma = (principal_vector / principal_vector.sum()).tolist()
        return alpha, beta, gamma

    def convert_to_grayscale(self) -> np.ndarray:
        """
        Converts the RGB image into grayscale using PCA-based projection.

        Returns
        -------
        np.ndarray
            Grayscale image matrix.
        """
        vect_mat = self._flatten_image()
        alpha, beta, gamma = self._compute_pca_components(vect_mat)
        grayscale_mat = (
            alpha * self.img_mat[:, :, 0] +
            beta * self.img_mat[:, :, 1] +
            gamma * self.img_mat[:, :, 2]
        )
        return grayscale_mat

    def show(self):
        """
        Displays the grayscale image using matplotlib.
        """
        grayscale_mat = self.convert_to_grayscale()
        plt.figure()
        plt.imshow(grayscale_mat, cmap='gray')
        plt.axis("off")
        plt.show()