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
gray = converter.convert_to_grayscale()

# ---------------------------------------------------
# Step 2: CLAHE Enhancement
# ---------------------------------------------------
clahe = CLAHEEnhancer(grayscale_mat=gray)
enhanced_rgb = clahe.convert_to_rgb()
enhanced_gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)

# ---------------------------------------------------
# Step 3: Gaussian Smoothing
# ---------------------------------------------------
blur_proc = GaussianBlurEnhancer(image_array=enhanced_rgb.astype(np.uint8))
blurred = blur_proc.apply_blur()
blur_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------
# Step 4: Canny Edge Mask
# ---------------------------------------------------
edges = cv2.Canny(blur_gray, 15, 45)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
clean_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

# ---------------------------------------------------
# Step 5: Border Cleanup
# ---------------------------------------------------
clean_mask[0:10, :] = 0
clean_mask[-10:, :] = 0
clean_mask[:, 0:10] = 0
clean_mask[:, -10:] = 0

# ---------------------------------------------------
# Step 6: Contour Detection + Feature Extraction
# ---------------------------------------------------
MIN_AREA = 2000
detector = ContourDetector(clean_mask, min_area=MIN_AREA)
detector.find_contours()
props = detector.compute_properties()  

overlay = detector.draw_big_contours(clean_mask)
detector.show(base_image=overlay)

big_contours = detector.get_big_contours()

print("Total detected:", len(big_contours))

# ---------------------------------------------------
# Build DataFrame with ALL 131 FEATURES
# ---------------------------------------------------

df = pd.DataFrame([
    {
        # -----------------------
        # 1. Geometry (17)
        # -----------------------
        "area": p["area"],
        "area_moment": p["area_moment"],
        "perimeter": p["perimeter"],
        "circularity": p["circularity"],
        "major_axis": p["major_axis"],
        "minor_axis": p["minor_axis"],
        "eccentricity": p["eccentricity"],
        "equivalent_diameter": p["equivalent_diameter"],
        "width_bb": p["width_bb"],
        "height_bb": p["height_bb"],
        "rectangularity": p["rectangularity"],
        "roundness": p["roundness"],
        "shape_factor": p["shape_factor"],
        "convexity": p["convexity"],
        "solidity": p["solidity"],
        "extent": p["extent"],
        "centroid_x": p["centroid"][0],
        "centroid_y": p["centroid"][1],

        # -----------------------
        # 2. Hu Moments (7)
        # -----------------------
        **{f"hu_{i+1}": p[f"hu_{i+1}"] for i in range(7)},

        # -----------------------
        # 3. Zernike Moments (25)
        # -----------------------
        **{f"z_{i}": p[f"z_{i}"] for i in range(25)},

        # -----------------------
        # 4. Color / Gray (8)
        # -----------------------
        "ratio_rg": p["ratio_rg"],
        "ratio_rb": p["ratio_rb"],
        "ratio_bg": p["ratio_bg"],
        "gray_mean": p["gray_mean"],
        "gray_std": p["gray_std"],
        "gray_skew": p["gray_skew"],
        "gray_kurt": p["gray_kurt"],
        "gray_entropy": p["gray_entropy"],

        # -----------------------
        # 5. Haralick (13)
        # -----------------------
        **{f"h_{i}": p[f"h_{i}"] for i in range(13)},

        # -----------------------
        # 6. LBP (54)
        # -----------------------
        **{f"lbp_{i}": p[f"lbp_{i}"] for i in range(54)},

        # -----------------------
        # 7. Fourier Descriptors (10)
        # -----------------------
        **{f"fourier_{i}": p[f"fourier_{i}"] for i in range(10)},
    }
    for p in props
])

df = df[df['area'] > MIN_AREA]

print(df)
show(df)

