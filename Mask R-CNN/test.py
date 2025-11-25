# from mask_r_cnn.transformConfig import TransformsConfig

# cfg = TransformsConfig(shape=(224, 224), flip=0.5)

# print(cfg)

import os
print(os.path.abspath("./Mask R-CNN/Species-3/train/_annotations.coco.json"))
print("Exists:", os.path.exists("./Mask R-CNN/Species-3/train/_annotations.coco.json"))
