
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from .architecture.vgg16 import VGG16, LateFusionVGG16
from torch.optim import Optimizer

def train_model(model, train_loader, test_loader, trainset, testset, device, num_epochs, optimizer: Optimizer):
    """
    Train the model.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        trainset (Dataset): Training dataset (for calculating accuracy)
        testset (Dataset): Test dataset (for calculating accuracy)
        device (torch.device): Device to run training on
        num_epochs (int): Number of training epochs
        lr (float): Learning rate
        
    Returns:
        tuple: (train_acc_list, test_acc_list) - accuracy lists for each epoch
    """

    train_acc_list = []
    test_acc_list = []
    total_loss_list = []
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        # Training phase
        model.train()
        train_correct = 0
        total_loss = 0
        
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            
            # Zero the gradients computed for each weight
            optimizer.zero_grad()
            
            # Forward pass your image through the network
            output = model(data)
            
            # Compute the loss
            loss = F.cross_entropy(output, target)
            total_loss += loss.item()
            
            # Backward pass through the network
            loss.backward()
            
            # Update the weights
            optimizer.step()
            
            # Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
        
        # Evaluation phase
        test_correct = 0
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)  # Keep target on same device
                output = model(data)
                predicted = output.argmax(1)
                test_correct += (target==predicted).sum().item()
        
        # Calculate accuracies
        train_acc = train_correct/len(trainset)
        test_acc = test_correct/len(testset)
        
        # Store accuracies for plotting
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        total_loss_list.append(total_loss)
        
        avg_loss = total_loss / len(train_loader)
        print("Epoch {}: Loss: {:.4f}, Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(
            epoch+1, avg_loss, test=100*test_acc, train=100*train_acc))

    return train_acc_list, test_acc_list, total_loss_list


def plot_training_results(train_acc_list, test_acc_list, total_loss_list, title):
    """
    Plot training and test accuracy curves.
    
    Args:
        train_acc_list (list): Training accuracies for each epoch
        test_acc_list (list): Test accuracies for each epoch
    """
    num_epochs = len(train_acc_list)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_acc_list, label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), test_acc_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.savefig(title + '_accuracy.png')

    # Plot total loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), total_loss_list, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.savefig(title + '_loss.png')


def evaluate_model(model, data_loader, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model (nn.Module): The model to evaluate
        data_loader (DataLoader): Data loader for evaluation
        device (torch.device): Device to run evaluation on
        
    Returns:
        float: Accuracy on the dataset
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predicted = output.argmax(1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return correct / total
