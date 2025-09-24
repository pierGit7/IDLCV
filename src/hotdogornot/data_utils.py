"""
Data loading utilities for Hotdog/Not Hotdog classification.
"""

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
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomVerticalFlip(), 
        transforms.RandomHorizontalFlip(), 
        transforms.GaussianBlur(3), 
        transforms.ToTensor()
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((size, size)), 
        transforms.ToTensor()
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
