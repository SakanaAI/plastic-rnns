import os
import numpy as np
import torch
import torchvision

from tasks.image_classification import ImageClassificationTask, ImageClassificationDataset, CrossAttentionImageClassificationModel

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CIFAR10Dataset(ImageClassificationDataset):
    """CIFAR-10 dataset for image classification."""

    DATASET_CONFIG = {
        'dir': os.path.join(_PROJECT_ROOT, 'data', 'cifar10'),
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616],
        'image_size': 32,
        'padding': 4,
        'num_channels': 3,
        'num_classes': 10,
        'use_horizontal_flip': True,
    }

    def _load_data(self, phase: str):
        os.makedirs(self.config['dir'], exist_ok=True)
        if phase == 'train':
            base_dataset = torchvision.datasets.CIFAR10(
                root=self.config['dir'],
                train=True,
                download=True
            )
            self.data = base_dataset.data[:45000]  # (N, 32, 32, 3) uint8
            self.labels = base_dataset.targets[:45000]
        elif phase == 'val':
            base_dataset = torchvision.datasets.CIFAR10(
                root=self.config['dir'],
                train=True,
                download=True
            )
            self.data = base_dataset.data[45000:]
            self.labels = base_dataset.targets[45000:]
        else:  # test
            base_dataset = torchvision.datasets.CIFAR10(
                root=self.config['dir'],
                train=False,
                download=True
            )
            self.data = base_dataset.data
            self.labels = base_dataset.targets


class ImageClassificationCIFAR10Task(ImageClassificationTask):
    """CIFAR-10 Image Classification Task with CNN backbone for cross-attention models."""
    DATASET_CLASS = CIFAR10Dataset

    @classmethod
    def build_model(cls, cfg, model_type):
        dc = cls.DATASET_CLASS.DATASET_CONFIG
        model = CrossAttentionImageClassificationModel(
            model_cfg=cfg.model, task_cfg=cfg.task, model_type=model_type,
            num_channels=dc['num_channels'], img_size=dc['image_size']
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, dc['num_channels'], dc['image_size'], dc['image_size'])
            model(dummy, states=model.get_initial_states(batch_size=1))
        return model

    def _create_model(self, cfg, model_type, num_channels, img_size):
        return CrossAttentionImageClassificationModel(
            model_cfg=cfg.model,
            task_cfg=cfg.task,
            model_type=model_type,
            num_channels=num_channels,
            img_size=img_size
        )
