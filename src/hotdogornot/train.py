"""
Training utilities for Hotdog/Not Hotdog classification.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from .models import VGG16

def train_model(model: VGG16, train_loader, test_loader, trainset, testset, device, num_epochs=10, lr=0.1):
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
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_acc_list = []
    test_acc_list = []

    for epoch in tqdm(range(num_epochs), unit='epoch'):
        #For each epoch
        model.train()
        train_correct = 0
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #Compute the loss
            loss = F.nll_loss(torch.log(output), target)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()
            
            #Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
        #Comput the test accuracy
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data = data.to(device)
            with torch.no_grad():
                output = model(data)
            predicted = output.argmax(1).cpu()
            test_correct += (target==predicted).sum().item()
        train_acc = train_correct/len(trainset)
        test_acc = test_correct/len(testset)
        print("Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(test=100*test_acc, train=100*train_acc))
    
        return train_acc_list, test_acc_list


def plot_training_results(train_acc_list, test_acc_list):
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
    plt.title('Training and Test Accuracy')
    plt.grid(True)
    plt.show()


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
