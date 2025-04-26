import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code/')))
from component.model import LRCN
from component.utils.dataset import VideoDataset
from component.utils.visualization import plot_metrics
from component.utils.utility import get_current_lr, save_best_model, evaluate_model
from torchvision import transforms
from dotenv import load_dotenv
import datetime
import json
import logging

# Load environment variables from .env file
load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging setup using logging module
logging.basicConfig(filename=os.getenv("LOG_FILE_PATH"),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set seed for reproducibility
torch.manual_seed(int(os.getenv("SEED", 27)))
logging.info("Retained old model, added bn and dropout, added lr scheduler (initial lr = 0.0001,stepsize = 6)")

# Data and Model Configurations
DATASET_DIR = os.getenv("DATASET_DIR")
CLASSES_LIST = json.loads(os.getenv("CLASSES_LIST"))

# HYPERPARAMS
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))
EPOCHS = int(os.getenv("EPOCHS", 30))
LR = float(os.getenv("LR", 0.1))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-4))
STEP_SIZE = int(os.getenv("STEP_SIZE", 10))
GAMMA = float(os.getenv("GAMMA", 0.1))
# TRAIN/VALIDATION SPLIT
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", 0.6))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", 0.2))


# Initialize dataset and dataloaders
dataset = VideoDataset(DATASET_DIR, CLASSES_LIST)
train_size = int(TRAIN_SPLIT * len(dataset))
val_size = int(VALIDATION_SPLIT * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Model, Loss, Optimizer Setup
model = LRCN(num_classes=len(CLASSES_LIST)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# Metrics lists to track the performance
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
best_test_acc = 0
best_model_state_dict = None

def train_and_evaluate():
    """
    Trains the model for the specified number of epochs and evaluates it on the train, validation, and test datasets.
    
    The function tracks and logs the training process, including loss, accuracy, and learning rate.
    It also saves the best model based on test accuracy and evaluates it on all datasets.

    Returns:
        None
    """
    global best_test_acc, best_model_state_dict

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for videos, labels in train_dl:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_losses.append(total_loss)
        train_accuracies.append(train_acc)

        # Test Step
        test_acc, test_loss = evaluate_model(model, test_dl, criterion, "Test")
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # Track best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state_dict = model.state_dict()

        # Update learning rate scheduler
        scheduler.step()

        current_lr = get_current_lr(optimizer)

        logging.info(f"Epoch: {epoch+1}, Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}, LR: {current_lr:.6f}")
        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, LR: {current_lr:.6f}")

    # Save the best model
    save_best_model(model, best_model_state_dict)

    # Load best model for final evaluation
    model.load_state_dict(best_model_state_dict)
    model.eval()

    # Evaluate performance on all datasets
    train_accuracy = evaluate_model(model, train_dl, criterion, "Train")[0]
    validation_accuracy = evaluate_model(model, val_dl, criterion, "Validation")[0]
    test_accuracy = evaluate_model(model, test_dl, criterion, "Test")[0]

    # Log Final Accuracies
    logging.info(f"Final Train Accuracy: {train_accuracy:.2f}%")
    logging.info(f"Final Validation Accuracy: {validation_accuracy:.2f}%")
    logging.info(f"Final Test Accuracy: {test_accuracy:.2f}%")

    #print
    print(f"Final Train Accuracy: {train_accuracy:.2f}%")
    print(f"Final Validation Accuracy: {validation_accuracy:.2f}%")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    # Plot metrics (accuracy, loss)
    plot_metrics(train_accuracies, test_accuracies, train_losses, test_losses)

# Call the training and evaluation function
if __name__ == "__main__":
    # train_and_evaluate()
    pass
