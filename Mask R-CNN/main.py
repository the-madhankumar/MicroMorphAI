from mask_r_cnn.transformConfig import TransformsConfig
from mask_r_cnn.customDataset import CocoSegmentationDataset
from mask_r_cnn.detection.engine import train_one_epoch, evaluate
from mask_r_cnn.model_builder import MaskRCNNModelBuilder

from torch.utils.data import DataLoader
import torch
import math


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

def main():
    print("Initializing Transform Config...")
    cfg = TransformsConfig(shape=(224, 224), flip=0.5)
    train_tf = cfg.build_transforms(train=True)
    test_tf = cfg.build_transforms(train=False)

    print("Loading Datasets...")
    train_dataset = CocoSegmentationDataset(
        root_dir='./Mask R-CNN/Species-3/train',
        annotation_file='./Mask R-CNN/Species-3/train/_annotations_fixed.coco.json',
        transforms=train_tf
    )

    val_dataset = CocoSegmentationDataset(
        root_dir='./Mask R-CNN/Species-3/valid',
        annotation_file='./Mask R-CNN/Species-3/valid/_annotations.coco.json',
        transforms=test_tf
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print("Initializing Model...")
    builder = MaskRCNNModelBuilder()
    device = builder.get_device()

    num_classes = len(train_dataset.coco.getCatIds()) + 1
    model = builder.build(num_classes=num_classes, device=device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    num_epochs = 10
    best_val_loss = math.inf
    patience = 100
    patience_counter = 0

    print("Training Started...\n")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Skip training if batch is empty
        if len(train_loader) == 0:
            print("[ERROR] Training loader is empty. Check dataset!")
            break

        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=25)
        val_stats = evaluate(model, val_loader, device=device)
        val_loss = val_stats["validation_loss"]

        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Model improved. Saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping.")
            break

    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
    print("Training Completed.")


if __name__ == "__main__":
    main()
