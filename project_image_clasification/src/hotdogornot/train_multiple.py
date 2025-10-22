"""
Main script to run the Hotdog/Not Hotdog classification training.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hotdogornot.architecture.vgg import VGG16
from hotdogornot.data_utils import get_dataloaders
from hotdogornot.train import train_model, plot_training_results
from hotdogornot.utils import setup_device, visualize_samples, count_parameters
from hotdogornot.architecture.resnet18 import ResNet18
import torch.optim as optim
from torchvision.models import vgg16, VGG16_Weights, vgg16_bn
import torch.nn as nn


def train_multiple():
    """Main function to run the training pipeline."""
    # Setup device
    device = setup_device()

    # Data loading parameters
    batch_size = 64
    size = 224

    # Get data loaders
    print("Loading data...")
    train_loader, test_loader, trainset, testset = get_dataloaders(
        batch_size=batch_size, size=size
    )

    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")

    # Visualize some samples
    print("Visualizing sample images...")
    # visualize_samples(train_loader)
    # Create model
    print("Creating VGG model...")

    # Training parameters
    num_epochs = 50
    lr = [1e-2, 1e-3, 1e-4]  # Increased learning rate for better convergence
    optimizer_map = {
        "SGD_lr_" + str(lr[2]): lr[2],
        "SGD_lr_" + str(lr[0]): lr[0],
        "SGD_lr_" + str(lr[1]): lr[1],
        "Adam_lr_" + str(lr[0]): lr[0],
        "Adam_lr_" + str(lr[1]): lr[1],
        "Adam_lr_" + str(lr[2]): lr[2],
        "RMSprop_lr_" + str(lr[0]): lr[0],
        "RMSprop_lr_" + str(lr[1]): lr[1],
        "RMSprop_lr_" + str(lr[2]): lr[2],
    }

    for key in optimizer_map.keys():

        model = vgg16_bn(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2)
        )        
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

        lr = optimizer_map[key]
        if "Adam" in key:
            optimizer = optim.Adam(model.parameters(), lr=optimizer_map[key])
            print(f"Using Adam optimizer with lr={optimizer_map[key]}")
        elif "SGD" in key:
            optimizer = optim.SGD(
                model.parameters(), lr=optimizer_map[key], momentum=0.9
            )
            print(f"Using SGD optimizer with lr={optimizer_map[key]}")
        elif "RMSprop" in key:
            optimizer = optim.RMSprop(model.parameters(), lr=optimizer_map[key])
            print(f"Using RMSprop optimizer with lr={optimizer_map[key]}")

        # Train the model
        print(f"Starting training for {num_epochs} epochs...")
        train_acc_list, test_acc_list, total_loss_list = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            trainset=trainset,
            testset=testset,
            device=device,
            num_epochs=num_epochs,
            optimizer=optimizer,
        )

        # Plot results
        print("Plotting training results...")
        plot_training_results(
            train_acc_list,
            test_acc_list,
            total_loss_list,
            title=f'{optimizer.__class__.__name__} lr={optimizer.param_groups[0]["lr"]}, VGG16',
        )

        # Save the trained model
        model_path = f'models/VGG16_hotdog.pth_{optimizer.__class__.__name__}, lr={optimizer.param_groups[0]["lr"]}, VGG16'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        from hotdogornot.utils import save_model

        save_model(model, model_path)

        print("Training completed!")


if __name__ == "__main__":
    train_multiple()
