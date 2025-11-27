import cv2
import torch
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI
from torchvision import transforms, models
from pydantic import BaseModel

app = FastAPI()

MASK_R_CNN_PATH = r"D:/projects/MicroMorph AI/Models/model_epoch_10.pth"
NUM_CLASSES = 12

class ImagePathModel(BaseModel):
    Image_Path: str

class Random(BaseModel):
    values: int

# RCNN CONFIGS
class RCNNResponse(BaseModel):
    message: str
    image_base64: str
    detected_class_name: str
    confidence_level: float

CLASS_NAMES = [
    "Alexandrium",
    "Cerataulina",
    "Ceratium",
    "Entomoneis",
    "Guinardia",
    "Hemiaulus",
    "Nitzschia",
    "Pinnularia",
    "Pleurosigma",
    "Prorocentrum",
    "UnknownClass"
]

MaskRCNN_model = models.detection.maskrcnn_resnet50_fpn(
    weights=None,
    num_classes=NUM_CLASSES
)

MaskRCNN_model.load_state_dict(
    torch.load(MASK_R_CNN_PATH, map_location=torch.device("cpu"), weights_only=False)
)

MaskRCNN_model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

def img_to_base64(img):
    pil_img = Image.fromarray(img)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

@app.get("/root")
async def root():
    return {"message": "Hello World"}

@app.post("/mask_r_cnn")
async def mask_r_cnn(image_path: ImagePathModel) -> RCNNResponse:
    img_path = image_path.Image_Path
    image_bgr = cv2.imread(img_path)

    if image_bgr is None:
        return {"error": "Image not found"}

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = transform(Image.fromarray(image_rgb)).unsqueeze(0)

    with torch.no_grad():
        predictions = MaskRCNN_model(image_tensor)

    masks = predictions[0]['masks']
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    threshold = 0.4
    overlay = image_rgb.copy()

    for i in range(len(masks)):
        if scores[i] < threshold:
            continue

        mask = masks[i, 0].mul(255).byte().cpu().numpy()
        mask_bool = mask > 127
        box = boxes[i].cpu().numpy().astype(int)
        class_name = CLASS_NAMES[labels[i] - 1]
        score = float(scores[i])
        color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()

        colored_mask = np.zeros_like(overlay)
        for c in range(3):
            colored_mask[:, :, c] = mask_bool * color[c]

        alpha = 0.4
        overlay = np.where(
            mask_bool[:, :, None],
            ((1 - alpha) * overlay + alpha * colored_mask).astype(np.uint8),
            overlay
        )

        x1, y1, x2, y2 = box
        cv2.putText(
            overlay,
            f"{class_name}: {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    result_base64 = img_to_base64(overlay)

    return {
        "message": "Mask R-CNN processed successfully",
        "image_base64": result_base64,
        "detected_class_name": class_name,
        "confidence_level": score,
    }

@app.post("/random_forest")
async def randomForest(data: Random):
    return {"received_value": data.values}
