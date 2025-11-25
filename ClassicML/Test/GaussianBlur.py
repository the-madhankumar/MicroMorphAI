import cv2
import numpy as np

# img = cv2.imread('D:\projects\Project MicroMorph AI\Images\TestImages\gaussianblur.webp')

# dst = np.empty_like(img) #create empty array the size of the image
# noise = cv2.randn(dst, (0,0,0), (20,20,20)) #add random img noise

# # Pass img through noise filter to add noise
# pup_noise = cv2.addWeighted(img, 0.5, noise, 0.5, 50) 

# # Blurring function; kernel=15, sigma=auto
# pup_blur = cv2.GaussianBlur(pup_noise, (15, 15), 0)

# cv2.imshow('Img', pup_blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows

import cv2
import numpy as np

class GaussianBlurEnhancerFixed:
    def __init__(self, image_array):
        self.img = image_array.astype(np.float32)  # always float32 internally

    def add_noise(self, mean=0, std=20):
        noise = np.random.normal(mean, std, self.img.shape).astype(np.float32)
        noisy_img = self.img + noise
        return noisy_img

    def apply_blur(self, kernel_size=(15,15), sigma=0):
        noisy_img = self.add_noise()
        blurred_img = cv2.GaussianBlur(noisy_img, kernel_size, sigma)
        # clip and convert to uint8 for display
        blurred_img_uint8 = np.clip(blurred_img, 0, 255).astype(np.uint8)
        return blurred_img_uint8

# Example usage
from PIL import Image
img = np.array(Image.open('D:/projects/Project MicroMorph AI/Images/TestImages/grayscale.webp'))

# Convert grayscale to 3 channels if needed
if len(img.shape) == 2:
    img = np.stack([img]*3, axis=-1)

enhancer = GaussianBlurEnhancerFixed(img)
blurred_img = enhancer.apply_blur()

cv2.imshow("Blurred Image", blurred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

