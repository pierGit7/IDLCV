"""
Utility functions for Hotdog/Not Hotdog classification.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def setup_device():
    """
    Setup device for training (GPU if available, else CPU).
    
    Returns:
        torch.device: The device to use for training
    """
    if torch.cuda.is_available():
        print("The code will run on GPU.")
        device = torch.device('cuda')
    else:
        print("The code will run on CPU.")
        device = torch.device('cpu')
    
    return device


def visualize_samples(data_loader, num_samples=21):
    """
    Visualize sample images from the data loader.
    
    Args:
        data_loader (DataLoader): Data loader containing the images
        num_samples (int): Number of samples to visualize
    """
    images, labels = next(iter(data_loader))
    
    plt.figure(figsize=(20, 10))
    print(f"Image shape: {images[0].shape}")
    
    for i in range(min(num_samples, len(images))):
        plt.subplot(5, 7, i+1)
        # Convert from (C, H, W) to (H, W, C) for matplotlib
        img = np.transpose(images[i].numpy(), (1, 2, 0))
        plt.imshow(img)
        plt.title(['hotdog', 'not hotdog'][labels[i].item()])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): The model to count parameters for
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, filepath):
    """
    Save model state dict to file.
    
    Args:
        model (nn.Module): Model to save
        filepath (str): Path to save the model
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device):
    """
    Load model state dict from file.
    
    Args:
        model (nn.Module): Model to load weights into
        filepath (str): Path to the saved model
        device (torch.device): Device to load the model on
        
    Returns:
        nn.Module: Model with loaded weights
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f"Model loaded from {filepath}")
    return model
