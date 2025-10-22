"""
architecture package
"""

from .vgg import VGG16
from .resnet18 import ResNet18

__all__ = [
    'VGG16',
    'ResNet18',
]