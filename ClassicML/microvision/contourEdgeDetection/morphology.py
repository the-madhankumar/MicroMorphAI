import cv2
import numpy as np

class MorphologyProcessor:
    """
    Applies morphological operations (erosion, dilation, opening, closing, gradient) on binary or edge images.
    """

    def __init__(self, image: np.ndarray, kernel_size: int = 0, kernel_shape: str = "cross"):
        """
        Initializes the MorphologyProcessor.

        Parameters
        ----------
        image : np.ndarray  
            Input image (usually a binary / edge image, 2D).
        kernel_size : int, optional  
            Size of the structuring element (must be odd).
        kernel_shape : str, optional  
            Shape of the structuring element: "rect", "ellipse", or "cross".
        """
        if len(image.shape) != 2:
            raise ValueError("MorphologyProcessor expects a 2D (single-channel) image")
        self.img = image
        self.kernel = self._get_kernel(kernel_size, kernel_shape)

    def _get_kernel(self, k: int, shape: str) -> np.ndarray:
        if shape == "rect":
            return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        elif shape == "ellipse":
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        elif shape == "cross":
            return cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
        else:
            raise ValueError("Unsupported kernel shape: choose from 'rect', 'ellipse', or 'cross'")

    def erode(self, iterations: int = 1) -> np.ndarray:
        return cv2.erode(self.img, self.kernel, iterations=iterations)

    def dilate(self, iterations: int = 1) -> np.ndarray:
        return cv2.dilate(self.img, self.kernel, iterations=iterations)

    def opening(self) -> np.ndarray:
        return cv2.morphologyEx(self.img, cv2.MORPH_OPEN, self.kernel)

    def closing(self) -> np.ndarray:
        return cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, self.kernel)

    def gradient(self) -> np.ndarray:
        return cv2.morphologyEx(self.img, cv2.MORPH_GRADIENT, self.kernel)

    def process(self, operations: list = ["opening", "closing", "gradient"]) -> np.ndarray:
        """
        Applies a sequence of morphological operations in order.

        Parameters
        ----------
        operations : list of str  
            List of operations to apply: "erode", "dilate", "opening", "closing", "gradient"

        Returns
        -------
        np.ndarray
            Processed image after applying operations in sequence.
        """
        img = self.img.copy()
        for op in operations:
            if op == "erode":
                img = cv2.erode(img, self.kernel, iterations=1)
            elif op == "dilate":
                img = cv2.dilate(img, self.kernel, iterations=1)
            elif op == "opening":
                img = cv2.morphologyEx(img, cv2.MORPH_OPEN, self.kernel)
            elif op == "closing":
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, self.kernel)
            elif op == "gradient":
                img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, self.kernel)
            else:
                raise ValueError(f"Unsupported operation: {op}")
        return img

    def show(self, result: np.ndarray, title: str = "Morphology Result"):
        """
        Displays the result image using OpenCV.

        Parameters
        ----------
        result : np.ndarray  
            Morphologically processed image to display.
        title : str, optional  
            Window title.
        """
        cv2.imshow(title, result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
