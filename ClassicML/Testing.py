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
    "D:/projects/Project MicroMorph AI/Images/TestImages/test8.png"
)
gray = converter.convert_to_grayscale()


# ---------------------------------------------------
# Step 2: Gaussian Blur
# ---------------------------------------------------
blur_proc = GaussianBlurEnhancer(image_array=gray.astype(np.uint8))
blurred = blur_proc.apply_blur()


# ---------------------------------------------------
# Step 3: Adaptive Threshold
# ---------------------------------------------------
blur_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

binary = cv2.adaptiveThreshold(
    blur_gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,
    2
)


# ---------------------------------------------------
# Step 4: Morphology
# ---------------------------------------------------
morph = MorphologyProcessor(binary, kernel_size=5, kernel_shape="ellipse")
clean_mask = morph.process(operations=["closing", "opening"])


# ---------------------------------------------------
# Step 5: Remove border noise
# ---------------------------------------------------
clean_mask[0:10, :] = 0
clean_mask[-10:, :] = 0
clean_mask[:, 0:10] = 0
clean_mask[:, -10:] = 0


# ---------------------------------------------------
# Step 6: Contour Detection
# ---------------------------------------------------
detector = ContourDetector(clean_mask, min_area=10000)

detector.find_contours()
props = detector.compute_properties()

overlay = detector.draw_big_contours(clean_mask)
detector.show(overlay, "Organism Contours")


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
        "aspect_ratio": p["aspect_ratio"],
        "solidity": p["solidity"],
        "extent": p["extent"],
        "compactness": p["compactness"],
        "hu1_log": p["hu_log"][0],
        "hu2_log": p["hu_log"][1],
        "hu3_log": p["hu_log"][2],
        "hu4_log": p["hu_log"][3],
        "hu5_log": p["hu_log"][4],
        "hu6_log": p["hu_log"][5],
        "hu7_log": p["hu_log"][6],
    }
    for p in filtered
])

print(df)
show(df)
