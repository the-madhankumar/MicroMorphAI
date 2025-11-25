import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO


class CocoSegmentationDataset(Dataset):
    """
    PyTorch dataset for COCO-style segmentation (for Mask R-CNN).

    Only includes images with at least one valid annotation.
    """

    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.transforms = transforms

        # Filter out images with no valid objects
        self.image_ids = []
        for img_id in self.coco.imgs.keys():
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            valid = False
            for ann in anns:
                if "bbox" in ann and "segmentation" in ann:
                    x, y, w, h = ann["bbox"]
                    if w > 0 and h > 0 and np.sum(self.coco.annToMask(ann)) > 0:
                        valid = True
                        break
            if valid:
                self.image_ids.append(img_id)

        if len(self.image_ids) == 0:
            raise RuntimeError("No valid images found in dataset!")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes, labels, masks, areas, iscrowd = [], [], [], [], []

        for ann in annotations:
            if "bbox" not in ann or "segmentation" not in ann:
                continue
            xmin, ymin, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            mask = self.coco.annToMask(ann)
            if mask.sum() == 0:
                continue

            xmax = xmin + w
            ymax = ymin + h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann["category_id"])
            masks.append(mask)
            areas.append(ann.get("area", float(mask.sum())))
            iscrowd.append(ann.get("iscrowd", 0))

        # Convert to tensors
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.tensor(np.stack(masks, axis=0), dtype=torch.uint8),
            "image_id": torch.tensor([image_id]),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target
