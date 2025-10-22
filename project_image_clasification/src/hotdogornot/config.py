"""
Configuration file for Hotdog/Not Hotdog classification.
"""

import os

class Config:
    """Configuration class containing all hyperparameters and paths."""
    
    # Data paths
    DATA_PATH = '../data/hotdog_nothotdog'
    MODEL_SAVE_PATH = '../models/resnet18_hotdog.pth'
    
    # Data parameters
    IMAGE_SIZE = 128
    BATCH_SIZE = 128
    NUM_CLASSES = 2
    
    # Training parameters
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    
    # Model parameters
    RESNET_LAYERS = [2, 2, 2, 2]  # For ResNet18
    
    # Visualization parameters
    VIS_SAMPLES = 21
    PLOT_FIGSIZE = (20, 10)
    
    @classmethod
    def get_absolute_path(cls, relative_path):
        """Convert relative path to absolute path."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(current_dir, relative_path))
    
    @classmethod
    def ensure_dir_exists(cls, path):
        """Ensure directory exists, create if it doesn't."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
