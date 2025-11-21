from microvision.imagePreprocessingPipline.grayScaleConverstion import PCAGrayscaleConverter
from microvision.imagePreprocessingPipline.gaussianBlur import GaussianBlurEnhancer

from microvision.contourEdgeDetection.morphology import MorphologyProcessor
from microvision.contourEdgeDetection.cannyEdge import CannyEdgeDetector
from microvision.contourEdgeDetection.contourDetection import ContourDetector

import cv2
import numpy as np
from PIL import Image
import pandas as pd


# ---------------------------------------------
# Step 1: PCA Grayscale
# ---------------------------------------------
converter = PCAGrayscaleConverter(
    "D:/projects/Project MicroMorph AI/Images/TestImages/Navicula.jpg"
)
gray = converter.convert_to_grayscale()
print("Gray:", gray.shape)


# ---------------------------------------------
# Step 2: Gaussian Blur
# ---------------------------------------------
blur_proc = GaussianBlurEnhancer(image_array=gray.astype(np.uint8))
blurred = blur_proc.apply_blur()
print("Blur:", blurred.shape)


# ---------------------------------------------
# Step 3: Otsu Threshold (Stable)
# ---------------------------------------------
blur_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(
    blur_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
print("Binary:", binary.shape)


# ---------------------------------------------
# Step 4: Morphology → clean mask
# ---------------------------------------------
morph = MorphologyProcessor(binary, kernel_size=3, kernel_shape="ellipse")
clean_mask = morph.process(operations=["opening"])   # removes dust

morph.show(clean_mask, "Clean Mask")


# ---------------------------------------------
# Step 5: Canny (Just for visualization – NOT for contour detection)
# ---------------------------------------------
canny = CannyEdgeDetector(clean_mask)
edges = canny.detect_edges(50, 150)
canny.show(edges, "Edges")


# ---------------------------------------------
# Step 6: Contours on CLEAN MASK (Not edges)
# ---------------------------------------------
detector = ContourDetector(edges, min_area=10000)

detector.find_contours()
props = detector.compute_properties()

# for p in props:
#     print(f"Area: {p['area']} | Centroid: {p['centroid']}")

stable = detector.draw_big_contours(clean_mask)
detector.show(stable, "Stable Contours")


# ---------------------------------------------
# Step 7: DataFrame Output
# ---------------------------------------------
filtered = detector.get_big_contours()

df = pd.DataFrame([
    {
        "area": p["area"],
        "perimeter": p["perimeter"],
        "centroid": p["centroid"]
    }
    for p in filtered
])

print(df)
