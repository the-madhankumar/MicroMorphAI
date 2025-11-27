import os
import cv2
import numpy as np
from tqdm import tqdm

from microvision.imagePreprocessingPipline.contrastEnhancement import CLAHEEnhancer
import chromadb

from uvision.embeddings import ImageEmbeddingEngine
from microvision.imagePreprocessingPipline.grayScaleConverstion import PCAGrayscaleConverter
from microvision.imagePreprocessingPipline.gaussianBlur import GaussianBlurEnhancer
from microvision.contourEdgeDetection.morphology import MorphologyProcessor
from microvision.contourEdgeDetection.contourDetection import ContourDetector


folders = [
    r"D:\projects\MicroMorph AI\Project MicroMorph AI\UnSeenVision\SpeciesSAM-2\test",
    r"D:\projects\MicroMorph AI\Project MicroMorph AI\UnSeenVision\SpeciesSAM-2\train",
    r"D:\projects\MicroMorph AI\Project MicroMorph AI\UnSeenVision\SpeciesSAM-2\valid"
]

engine = ImageEmbeddingEngine()
client = chromadb.PersistentClient(
    path=r"D:/projects/MicroMorph AI/Project MicroMorph AI/ModelSync/chroma_storage"
)
collection = client.get_or_create_collection("species_embeddings")

records = []

def get_target_from_filename(filename):
    return filename.split("_")[0]


def process_image(imagePath):
    # Step 1: PCA Grayscale
    converter = PCAGrayscaleConverter(imagePath)
    gray = converter.convert_to_grayscale()

    # Step 2: CLAHE Contrast Enhancement
    clahe = CLAHEEnhancer(grayscale_mat=gray)
    enhanced_rgb = clahe.convert_to_rgb()
    enhanced_gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)

    # Step 3: Gaussian Blur
    blur_proc = GaussianBlurEnhancer(image_array=enhanced_gray.astype(np.uint8))
    blurred = blur_proc.apply_blur()
    blur_gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Step 4: Adaptive Threshold
    binary = cv2.adaptiveThreshold(
        blur_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 2
    )

    # Step 5: Morphology
    morph = MorphologyProcessor(binary, kernel_size=5, kernel_shape="ellipse")
    clean_mask = morph.process(operations=["closing", "opening"])

    # Step 6: Remove Border Noise
    clean_mask[0:10, :] = 0
    clean_mask[-10:, :] = 0
    clean_mask[:, 0:10] = 0
    clean_mask[:, -10:] = 0

    # Step 7: Contour Detection
    detector = ContourDetector(clean_mask, min_area=8000)
    detector.find_contours()
    detector.compute_properties()

    # ❌ Removed overlay + show (fixes cv2 error)
    # overlay = detector.draw_big_contours(clean_mask)
    # detector.show(overlay, "Organism Contours")

    return detector.get_big_contours()


def store_to_chroma(imagePath, metadata, embedding):
    collection.add(
        ids=[os.path.basename(imagePath)],
        embeddings=[embedding.tolist()],
        metadatas=[metadata],
        documents=[imagePath]
    )


def create_metadata(p):
    return {
        "area": p["area"],
        "perimeter": p["perimeter"],
        "centroid": str(p["centroid"]),
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


# 🔥 Add tqdm progress bar
all_files = []
for folder in folders:
    for f in os.listdir(folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            all_files.append(os.path.join(folder, f))

for path in tqdm(all_files, desc="Processing images"):
    file = os.path.basename(path)

    contours = process_image(path)
    if len(contours) == 0:
        continue

    emb = engine.generate_embeddings_from_image(path)

    for p in contours:
        metadata = create_metadata(p)
        metadata["target"] = get_target_from_filename(file)
        metadata["image"] = path

        store_to_chroma(path, metadata, emb)
        records.append(metadata)

print("Completed. Stored embeddings + metadata without approval.") 
