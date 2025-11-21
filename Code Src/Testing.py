from microvision.imagePreprocessingPipline.grayScaleConverstion import PCAGrayscaleConverter
from microvision.imagePreprocessingPipline.gaussianBlur import GaussianBlurEnhancer

from microvision.contourEdgeDetection.morphology import MorphologyProcessor
from microvision.contourEdgeDetection.contourDetection import ContourDetector

import cv2
import numpy as np
import pandas as pd
from pandasgui import show


# ---------------------------------------------------
# Step 1: PCA Grayscale
# ---------------------------------------------------
converter = PCAGrayscaleConverter(
    "D:/projects/Project MicroMorph AI/Images/TestImages/test6.jpeg"
)
gray = converter.convert_to_grayscale()


# ---------------------------------------------------
# Step 2: Gaussian Blur
# ---------------------------------------------------
blur_proc = GaussianBlurEnhancer(image_array=gray.astype(np.uint8))
blurred = blur_proc.apply_blur()


# ---------------------------------------------------
# Step 3: Adaptive Threshold (More stable than Otsu)
# ---------------------------------------------------
blur_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

binary = cv2.adaptiveThreshold(
    blur_gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,  # block size
    2
)


# ---------------------------------------------------
# Step 4: Morphology (closing + opening)
# ---------------------------------------------------
morph = MorphologyProcessor(binary, kernel_size=5, kernel_shape="ellipse")

clean_mask = morph.process(operations=["closing", "opening"])


# ---------------------------------------------------
# Step 5: Remove image border to prevent false contour
# ---------------------------------------------------
clean_mask[0:10, :] = 0
clean_mask[-10:, :] = 0
clean_mask[:, 0:10] = 0
clean_mask[:, -10:] = 0


# ---------------------------------------------------
# Step 6: Contours on CLEAN MASK (not Canny)
# ---------------------------------------------------
detector = ContourDetector(clean_mask, min_area=10000)

detector.find_contours()
props = detector.compute_properties()

stable = detector.draw_big_contours(clean_mask)
detector.show(stable, "Organism Contours")


# ---------------------------------------------------
# Step 7: DataFrame Output
# ---------------------------------------------------
filtered = detector.get_big_contours()

df = pd.DataFrame([
    {
        "area": p["area"],
        "perimeter": p["perimeter"],
        "centroid": p["centroid"],
        "circularity": p["circularity"],
        "eccentricity": p["eccentricity"],
        "major_axis": p["major_axis"],
        "minor_axis": p["minor_axis"],
        "hu1": p["hu1"]
    }
    for p in filtered
])

print(df)
show(df)
