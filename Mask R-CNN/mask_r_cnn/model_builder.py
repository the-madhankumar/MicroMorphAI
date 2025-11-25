import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class MaskRCNNModelBuilder:
    """
    Builds and configures a Mask R-CNN model with a ResNet-50 FPN backbone.

    This class:
    - Loads a pretrained Mask R-CNN model.
    - Replaces the classification and mask prediction heads based on the number of dataset classes.
    - Moves the model to the desired device.

    Methods
    -------
    build(num_classes: int, device: torch.device) -> MaskRCNN
        Returns a fully configured Mask R-CNN model ready for training.
    """

    def __init__(self):
        pass

    def get_device(self):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def build(self, num_classes: int, device: torch.device) -> MaskRCNN:
        """
        Creates and configures a Mask R-CNN model.

        Parameters
        ----------
        num_classes : int
            Number of target classes including background.
        device : torch.device
            Device on which the model should run (e.g., CPU or CUDA).

        Returns
        -------
        MaskRCNN
            A Mask R-CNN model configured with custom predictors.
        """
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden, num_classes)

        return model.to(device)