import cv2
from microvision.imagePreprocessingPipline.grayScaleConverstion import PCAGrayscaleConverter
from microvision.imagePreprocessingPipline.contrastEnhancement import CLAHEEnhancer

converter = PCAGrayscaleConverter("D:/projects/Project MicroMorph AI/Images/TestImages/grayscale.webp")
grayscale_value = converter.convert_to_grayscale()

enhancer = CLAHEEnhancer(grayscale_mat=grayscale_value)
enhancer.show(title="CLAHE Output", axis="off")