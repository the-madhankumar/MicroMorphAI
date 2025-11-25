"""
transformConfig.py

This module defines a configuration class for creating image transformation
pipelines used in Mask R-CNN training and evaluation.

The transformations are applied to both images and targets (bounding boxes,
masks, labels) using TorchVision transforms.
"""

import torchvision.transforms as T


class TransformsConfig:
    """
    A configuration class to build training and validation transforms.

    Args:
        shape (tuple): Resize dimension (height, width).
        flip (float): Probability for random horizontal flip used in training.

    Methods:
        build_transforms(train=True):
            Returns a torchvision Compose() object containing the required
            transforms for training or evaluation.
    """

    def __init__(self, shape=(224, 224), flip=0.5):
        self.shape = shape
        self.flip = flip

    def build_transforms(self, train=True):
        """
        Build a transformation pipeline for Mask R-CNN.

        Parameters:
            train (bool): If True, includes augmentation steps.

        Returns:
            torchvision.transforms.Compose: A callable transform object.
        """

        transforms = []

        # Resize (image & target)
        transforms.append(T.Resize(self.shape))

        # Apply only during training
        if train:
            transforms.append(T.RandomHorizontalFlip(self.flip))

        # Convert image to tensor
        transforms.append(T.ToTensor())

        return T.Compose(transforms)
