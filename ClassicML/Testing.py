from microvision.imagePreprocessingPipline.contrastEnhancement import CLAHEEnhancer
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
    "D:/projects/MicroMorph AI/Project MicroMorph AI/UnSeenVision/SpeciesSAM-2/train/Cerataulina_41_png.rf.be8e4f525adc1008643d2904512cc89c.jpg"
)
gray = converter.convert_to_grayscale()  # shape: (H,W)


# ---------------------------------------------------
# Step 2: CLAHE Enhancement (NEW)
# ---------------------------------------------------
clahe = CLAHEEnhancer(grayscale_mat=gray)
enhanced_rgb = clahe.convert_to_rgb()
enhanced_gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------
# Step 3: Smooth image
# ---------------------------------------------------
blur_proc = GaussianBlurEnhancer(image_array=enhanced_rgb.astype(np.uint8))
blurred = blur_proc.apply_blur()
blur_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------
# Step 4: Edge-based Mask (Better than adaptive threshold)
# ---------------------------------------------------
edges = cv2.Canny(blur_gray, 15, 45)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
clean_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)


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
detector = ContourDetector(clean_mask, min_area=2000)

detector.find_contours()
props = detector.compute_properties()

overlay = detector.draw_big_contours(clean_mask)
detector.show(base_image=overlay)


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
