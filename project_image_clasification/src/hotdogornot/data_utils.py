"""
Data loading utilities for Hotdog/Not Hotdog classification.
"""

from numpy import size
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from .dataset import Hotdog_NotHotdog


def get_transforms(size=128):
    """
    Get train and test transforms.
    
    Args:
        size (int): Target image size
        
    Returns:
        tuple: (train_transform, test_transform)
    """
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])

    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        normalize
    ])

    
    test_transform = transforms.Compose([
        transforms.Resize((size, size)), 
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, test_transform


def get_dataloaders(data_path='data/hotdog_nothotdog', batch_size=128, size=128):
    """
    Get train and test data loaders.
    
    Args:
        data_path (str): Path to the dataset
        batch_size (int): Batch size for data loaders
        size (int): Target image size
        
    Returns:
        tuple: (train_loader, test_loader, trainset, testset)
    """
    train_transform, test_transform = get_transforms(size)
    
    trainset = Hotdog_NotHotdog(train=True, transform=train_transform, data_path=data_path)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    testset = Hotdog_NotHotdog(train=False, transform=test_transform, data_path=data_path)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, trainset, testset
