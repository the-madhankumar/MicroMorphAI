from microvision.imagePreprocessingPipline.grayScaleConverstion import PCAGrayscaleConverter
from microvision.imagePreprocessingPipline.contrastEnhancement import CLAHEEnhancer
from microvision.imagePreprocessingPipline.gaussianBlur import GaussianBlurEnhancer
from microvision.imagePreprocessingPipline.imageThresholding import AdaptiveThreshold

from microvision.contourEdgeDetection.cannyEdgeDetection import CannyEdgeDetector
import cv2
import numpy as np
from PIL import Image

# -------------------------
# Step 1: PCA Grayscale
# -------------------------
converter = PCAGrayscaleConverter("D:/projects/Project MicroMorph AI/Images/TestImages/Navicula.jpg")
grayscale_value = converter.convert_to_grayscale()
print("After Grayscale convert : ", grayscale_value.shape, grayscale_value.dtype, 
      grayscale_value.min(), grayscale_value.max())

# -------------------------
# Step 2: CLAHE Enhancement
# -------------------------
enhancer = CLAHEEnhancer(grayscale_mat=grayscale_value)
enhanced_img = enhancer.convert_to_rgb()  # ensure 3-channel output
print("After CLAHE enhancement : ", enhanced_img.shape, enhanced_img.dtype, 
      enhanced_img.min(), enhanced_img.max())

# Ensure dtype is uint8 for OpenCV
enhanced_img_uint8 = np.clip(enhanced_img, 0, 255).astype(np.uint8)
print("After CLAHE enhancement after uint8 : ", enhanced_img_uint8.shape, enhanced_img_uint8.dtype, 
      enhanced_img_uint8.min(), enhanced_img_uint8.max())

# -------------------------
# Step 3: Gaussian Blur + Noise
# -------------------------
blur_enhancer = GaussianBlurEnhancer(image_array=enhanced_img_uint8)
blurred_img = blur_enhancer.apply_blur()
print("After enhanced with Gaussian : ", blurred_img.shape, blurred_img.dtype, 
      blurred_img.min(), blurred_img.max())

thresholding = AdaptiveThreshold(blurred_img)

# Adaptive mean threshold
mean_binary = thresholding.mean_threshold(block_size=15, c=2)
thresholding.show(mean_binary, title="Mean Adaptive Threshold")

# Adaptive Gaussian threshold
gaussian_binary = thresholding.gaussian_threshold(block_size=15, c=2)
thresholding.show(gaussian_binary, title="Gaussian Adaptive Threshold")