import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from torch.optim import Optimizer


def train_model(model, train_loader, val_loader, test_loader,
                trainset, valset, testset, device, num_epochs, optimizer: Optimizer,
                early_stopping_patience=5, class_names=None):
    """
    Train a model with validation, early stopping, and confusion matrix.

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        trainset, valset, testset (Dataset): Corresponding datasets
        device (torch.device): Training device
        num_epochs (int): Maximum number of epochs
        optimizer (torch.optim.Optimizer): Optimizer
        early_stopping_patience (int): Early stopping patience
        class_names (list[str]): Optional list of class names for confusion matrix

    Returns:
        model (nn.Module): Best model (based on validation)
        history (dict): Accuracy and loss logs
        cm (ndarray): Confusion matrix on test set
    """

    model.to(device)

    history = {
        'train_acc': [],
        'val_acc': [],
        'test_acc': [],
        'loss': []
    }

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        # ---------------- TRAIN ----------------
        model.train()
        total_loss = 0
        train_correct = 0

        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = output.argmax(1)
            train_correct += (preds == target).sum().item()

        train_acc = train_correct / len(trainset)
        avg_loss = total_loss / len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                preds = output.argmax(1)
                val_correct += (preds == target).sum().item()
        val_acc = val_correct / len(valset)

        # ---------------- TEST ----------------
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                preds = output.argmax(1)
                test_correct += (preds == target).sum().item()
        test_acc = test_correct / len(testset)

        # ---------------- LOGGING ----------------
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        history['loss'].append(avg_loss)

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Loss={avg_loss:.4f}, "
              f"Train={train_acc*100:.2f}%, "
              f"Val={val_acc*100:.2f}%, "
              f"Test={test_acc*100:.2f}%")

        # ---------------- EARLY STOPPING ----------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1} "
                      f"(best val acc: {best_val_acc*100:.2f}%)")
                break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    print(f"✅ Training complete. Best validation accuracy: {best_val_acc*100:.2f}%")

    # ---------------- CONFUSION MATRIX ----------------
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, class_names, title="Confusion Matrix (Test Set)")

    return model, history, cm


def plot_training_results(history, title):
    """Plot accuracy and loss curves."""
    num_epochs = len(history['train_acc'])
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), history['train_acc'], label='Train')
    plt.plot(range(1, num_epochs + 1), history['val_acc'], label='Validation')
    plt.plot(range(1, num_epochs + 1), history['test_acc'], label='Test')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title}_accuracy.png")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), history['loss'], label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title}_loss.png")


def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix"):
    """Plot a confusion matrix using sklearn and matplotlib."""
    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names if class_names else None)
    disp.plot(cmap=plt.cm.Blues, colorbar=False)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()


def evaluate_model(model, data_loader, device):
    """Evaluate accuracy on a given dataset."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            preds = outputs.argmax(1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / total
