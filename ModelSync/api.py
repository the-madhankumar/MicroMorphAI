import cv2
import torch
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI
from torchvision import transforms, models
from pydantic import BaseModel
from typing import List
import chromadb

from uvision.embeddings import ImageEmbeddingEngine

# ----------------------------
# SAM 2.1 IMPORTS
# ----------------------------
# from sam2.build_sam import build_sam2
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# import supervision as sv


app = FastAPI()

# ===========================================================
# CONFIG
# ===========================================================
MASK_R_CNN_PATH = r"D:/projects/MicroMorph AI/Models/model_epoch_10.pth"
NUM_CLASSES = 12

# SAM_CHECKPOINT = r"D:/projects/MicroMorph AI/SAM/checkpoints/checkpoint.pt"
# SAM_CONFIG = r"D:/projects/MicroMorph AI/SAM/configs/sam2.1/sam2.1_hiera_b+.yaml"

# ===========================================================
# INPUT MODELS
# ===========================================================
class ImagePathModel(BaseModel):
    Image_Path: str

class EmbeddingModel(BaseModel):
    Image_Path: str
    n_results: int

class Random(BaseModel):
    values: int

class RCNNResponse(BaseModel):
    message: str
    image_base64: str
    detected_class_name: str
    confidence_level: float

class SAMResponse(BaseModel):
    message: str
    image_base64: str
    total_masks: int


# ===========================================================
# CLASS NAMES (adjust to your dataset)
# ===========================================================
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


# ===========================================================
# LOAD MASK R-CNN
# ===========================================================
MaskRCNN_model = models.detection.maskrcnn_resnet50_fpn(
    weights=None,
    num_classes=NUM_CLASSES
)
MaskRCNN_model.load_state_dict(
    torch.load(MASK_R_CNN_PATH, map_location=torch.device("cpu"), weights_only=False)
)
MaskRCNN_model.eval()

transform = transforms.Compose([transforms.ToTensor()])


# ===========================================================
# CHROMADB INIT
# ===========================================================
client = chromadb.PersistentClient(
    path=r"D:/projects/MicroMorph AI/Project MicroMorph AI/ModelSync/chroma_storage"
)
collection = client.get_or_create_collection("species_embeddings")

engine = ImageEmbeddingEngine()


# ===========================================================
# UTILS
# ===========================================================
def img_to_base64(img):
    pil_img = Image.fromarray(img)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# ===========================================================
# LOAD SAM 2.1 MODEL
# ===========================================================
# print("Loading SAM 2.1 model...")

# sam2_model = build_sam2(SAM_CONFIG, SAM_CHECKPOINT, device="cuda")
# sam2_mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

# mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

# print("SAM Model Loaded Successfully!")


# ===========================================================
# ROUTES
# ===========================================================

@app.get("/")
async def root():
    return {"message": "API working successfully"}


# -----------------------------------------------------------
# MASK R-CNN
# -----------------------------------------------------------
@app.post("/mask_r_cnn", response_model=RCNNResponse)
async def mask_r_cnn(image_path: ImagePathModel):

    image_bgr = cv2.imread(image_path.Image_Path)
    if image_bgr is None:
        return {
            "message": "Image not found",
            "image_base64": "",
            "detected_class_name": "",
            "confidence_level": 0.0
        }

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

    best_class = ""
    best_score = 0.0

    for i in range(len(masks)):
        if scores[i] < threshold:
            continue

        mask = masks[i, 0].mul(255).byte().cpu().numpy()
        mask_bool = mask > 127
        box = boxes[i].cpu().numpy().astype(int)
        class_name = CLASS_NAMES[labels[i] - 1]
        score = float(scores[i])

        if score > best_score:
            best_class = class_name
            best_score = score

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
        "detected_class_name": best_class,
        "confidence_level": best_score
    }


# -----------------------------------------------------------
# RANDOM FOREST MOCK
# -----------------------------------------------------------
@app.post("/random_forest")
async def random_forest(data: Random):
    return {"received_value": data.values}



# -----------------------------------------------------------
# IMAGE EMBEDDING SEARCH
# -----------------------------------------------------------
@app.post("/embedding")
async def embed_similarity(data: EmbeddingModel):

    emb = engine.generate_embeddings_from_image(data.Image_Path)

    response = collection.query(
        query_embeddings=[emb],
        n_results=data.n_results
    )

    return response



# -----------------------------------------------------------
# SAM 2.1 PREDICTION ROUTE
# -----------------------------------------------------------
# @app.post("/sam_predict")
# async def sam_predict(image_path: ImagePathModel):

#     image_bgr = cv2.imread(image_path.Image_Path)
#     if image_bgr is None:
#         return {
#             "message": "Image not found",
#             "image_base64": "",
#             "total_masks": 0,
#             "detections": []
#         }

#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

#     sam_result = sam2_mask_generator.generate(image_rgb)
#     detections = sv.Detections.from_sam(sam_result=sam_result)

#     annotated_image = image_rgb.copy()
#     annotated_image = mask_annotator.annotate(annotated_image, detections=detections)

#     mask_class_outputs = []

#     for idx, mask in enumerate(detections.mask):

#         mask_bool = mask.astype(bool)

#         masked_region = image_rgb.copy()
#         masked_region[~mask_bool] = 0

#         tensor_img = transform(Image.fromarray(masked_region)).unsqueeze(0)

#         with torch.no_grad():
#             prediction = MaskRCNN_model(tensor_img)

#         scores = prediction[0]["scores"].cpu().numpy()
#         labels = prediction[0]["labels"].cpu().numpy()

#         if len(scores) == 0:
#             best_class_name = "UnknownClass"
#             best_score = 0.0
#         else:
#             best_idx = np.argmax(scores)
#             best_class_name = CLASS_NAMES[labels[best_idx] - 1]
#             best_score = float(scores[best_idx])

#         box = prediction[0]["boxes"][best_idx].cpu().numpy().tolist() if len(scores) > 0 else []

#         mask_class_outputs.append({
#             "mask_id": idx,
#             "class_name": best_class_name,
#             "confidence": best_score,
#             "bounding_box": box
#         })

#     img64 = img_to_base64(annotated_image)

#     return {
#         "message": "SAM segmentation + classification successful",
#         "image_base64": img64,
#         "total_masks": len(mask_class_outputs),
#         "detections": mask_class_outputs
#     }
