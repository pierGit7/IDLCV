"""
Main script to run the Hotdog/Not Hotdog classification training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hotdogornot.models import VGG16
from hotdogornot.data_utils import get_dataloaders
from hotdogornot.train import train_model, plot_training_results
from hotdogornot.utils import setup_device, visualize_samples, count_parameters


def main():
    """Main function to run the training pipeline."""
    # Setup device
    device = setup_device()
    
    # Data loading parameters
    batch_size = 128
    size = 128
    
    # Get data loaders
    print("Loading data...")
    train_loader, test_loader, trainset, testset = get_dataloaders(
        batch_size=batch_size, 
        size=size
    )
    
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    
    # Visualize some samples
    print("Visualizing sample images...")
    visualize_samples(train_loader)
    
    # Create model
    print("Creating VGG16 model...")
    model = VGG16()
    model.to(device)
    
    print(f"Model has {count_parameters(model):,} trainable parameters")
    
    # Test model with a batch
    data = next(iter(train_loader))[0].to(device)
    try:
        output = model(data)
        print(f"Model test successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"Model test failed: {e}")
        return
    
    # Training parameters
    num_epochs = 10
    lr = 1e-4
    
    # Train the model
    print(f"Starting training for {num_epochs} epochs...")
    train_acc_list, test_acc_list = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        trainset=trainset,
        testset=testset,
        device=device,
        num_epochs=num_epochs,
        lr=lr
    )
    
    # Plot results
    print("Plotting training results...")
    # plot_training_results(train_acc_list, test_acc_list)
    
    # Save the trained model
    model_path = '../../models/resnet18_hotdog.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    from hotdogornot.utils import save_model
    save_model(model, model_path)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
