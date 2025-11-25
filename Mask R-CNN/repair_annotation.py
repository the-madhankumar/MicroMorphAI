import json
import os

def is_valid_polygon(segmentation):
    # Check if segmentation is a list
    if not isinstance(segmentation, list):
        return False

    if len(segmentation) == 0:
        return False

    # Each polygon must be list of numbers
    for poly in segmentation:
        if not isinstance(poly, list):
            return False
        if len(poly) < 6:           # minimum 3 points (x,y,x,y,x,y)
            return False
        if len(poly) % 2 != 0:      # must be even count
            return False

    return True


def repair_coco_annotations(input_json, output_json):
    print("Reading input file...")

    with open(input_json, "r") as f:
        coco = json.load(f)

    valid_annotations = []
    removed = 0

    # Map for image sizes
    image_sizes = {
        img['id']: (img['width'], img['height'])
        for img in coco.get("images", [])
    }

    print("Checking annotations...")

    for ann in coco["annotations"]:
        seg = ann.get("segmentation", None)
        img_id = ann.get("image_id")

        # Skip missing segmentation
        if seg is None:
            removed += 1
            continue

        # If segmentation is RLE dictionary
        if isinstance(seg, dict):
            if "counts" not in seg or "size" not in seg:
                removed += 1
                continue
            else:
                valid_annotations.append(ann)
                continue

        # Validate polygon format
        if not is_valid_polygon(seg):
            removed += 1
            continue

        # image_id must exist
        if img_id not in image_sizes:
            removed += 1
            continue

        valid_annotations.append(ann)

    # Save output
    coco["annotations"] = valid_annotations

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)

    print("Repair completed successfully.")
    print("Valid annotations:", len(valid_annotations))
    print("Removed annotations:", removed)
    print("Output saved to:", output_json)


if __name__ == "__main__":
    repair_coco_annotations(
        input_json="./Mask R-CNN/Species-3/train/_annotations.coco.json",
        output_json="./Mask R-CNN/Species-3/train/_annotations_fixed.coco.json"
    )
