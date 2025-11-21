import cv2
import numpy as np

class CannyEdgeDetector:
    """
    Applies Canny edge detection on a binary or grayscale image.
    """

    def __init__(self, image: np.ndarray):
        """
        Initializes the detector with an image.

        Parameters
        ----------
        image : np.ndarray
            Grayscale or binary input image (2D or 3D). If 3D, it will be converted to grayscale.
        """
        if len(image.shape) == 3:
            self.img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.img = image

    def detect_edges(self, low_threshold: int = 50, high_threshold: int = 150, aperture_size: int = 3, use_L2gradient: bool = False) -> np.ndarray:
        """
        Runs Canny edge detection.

        Parameters
        ----------
        low_threshold : int
            Lower bound for hysteresis thresholding.
        high_threshold : int
            Upper bound for hysteresis thresholding.
        aperture_size : int, optional
            Aperture size for Sobel operator used internally.
        use_L2gradient : bool, optional
            Whether to use a more accurate L2 norm for gradient magnitude.

        Returns
        -------
        np.ndarray
            Binary edge map: edges = 255, non-edges = 0.
        """
        edges = cv2.Canny(self.img, low_threshold, high_threshold, apertureSize=aperture_size, L2gradient=use_L2gradient)
        return edges

    def show(self, edges: np.ndarray, title: str = "Canny Edges"):
        """
        Displays the edge map.

        Parameters
        ----------
        edges : np.ndarray
            The binary edge map from detect_edges().
        title : str
            Window title when showing with OpenCV.
        """
        cv2.imshow(title, edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, edges: np.ndarray, path: str = "edges.png"):
        """
        Saves the edge map to file.

        Parameters
        ----------
        edges : np.ndarray
            The binary edge map from detect_edges().
        path : str
            Path where to save the edges image.
        """
        cv2.imwrite(path, edges)
