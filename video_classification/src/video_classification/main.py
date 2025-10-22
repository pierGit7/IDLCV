"""
Main script to run the Hotdog/Not Hotdog classification training.
"""

import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_classification.architecture.vgg16 import VGG16, LateFusionVGG16
from datasets import get_frame_loader
from video_classification.train import train_model, plot_training_results
import torch.optim as optim
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

def main():
    """Main function to run the training pipeline."""
    # Setup device
    device = setup_device()
    
    # Data loading parameters
    batch_size = 64
    size = 128
    
    # Get data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_frame_loader(
    root_dir='video_classification/data/ufc10',
    batch_size=8,
    use_video=True,          # True = use FrameVideoDataset, False = FrameImageDataset
    stack_frames=True,       # Only matters if use_video=True
    n_workers=4
    )
    print(train_loader.dataset.shape)
    
    
    # Visualize some samples
    print("Visualizing sample images...")
    # visualize_samples(train_loader)
    # Create model
    print("Creating VGG model...")
    model = LateFusionVGG16(num_classes=10)
    model.to(device)
    
    # Test model with a batch
    data = next(iter(train_loader))[0].to(device)
    try:
        output = model(data)
        print(f"Model test successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"Model test failed: {e}")
        return
    
    # Training parameters
    num_epochs = 50
    lr = 1e-4  # Increased learning rate for better convergence
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Train the model
    print(f"Starting training for {num_epochs} epochs...")
    train_acc_list, test_acc_list = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        trainset=train_dataset,
        testset=test_dataset,
        device=device,
        num_epochs=num_epochs,
        optimizer=optimizer
    )
    
    # Plot results
    print("Plotting training results...")
    plot_training_results(train_acc_list, test_acc_list)
    
    
    print("Training completed!")


if __name__ == "__main__":
    main()
