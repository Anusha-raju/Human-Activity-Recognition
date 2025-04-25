import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import LRCN
from utils.dataset import VideoDataset
from utils.visualization import plot_metrics
from utils.utility import get_current_lr, evaluate_performance
from torchvision import transforms
from dotenv import load_dotenv
import datetime
import json
import logging

# Load environment variables from .env file
load_dotenv()

# Logging setup using logging module
logging.basicConfig(filename=os.getenv("LOG_FILE_PATH"),
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set seed for reproducibility
torch.manual_seed(int(os.getenv("SEED", 27)))

# Data and Model Configurations
DATASET_DIR = os.getenv("DATASET_DIR")
CLASSES_LIST = json.loads(os.getenv("CLASSES_LIST"))

#HYPERPARAMS
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))
EPOCHS = int(os.getenv("EPOCHS", 30))
LR = float(os.getenv("LR", 0.1))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-4))
STEP_SIZE = int(os.getenv("STEP_SIZE", 10))
GAMMA = float(os.getenv("GAMMA", 0.1))
#TRAIN/VALIDATION SPLIT
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", 0.6))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", 0.2))
#MODEL_DIR TO SAVED MODEL
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# Transformation on train dataset (Data Augmentation)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
])

# Initialize dataset and dataloaders
dataset = VideoDataset(DATASET_DIR, CLASSES_LIST, transform=train_transform)
train_size = int(TRAIN_SPLIT * len(dataset))
val_size = int(VALIDATION_SPLIT * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Model, Loss, Optimizer Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LRCN(num_classes=len(CLASSES_LIST)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# Metrics lists to track the performance
train_losses, val_losses, test_losses = [], [], []
train_accuracies, val_accuracies, test_accuracies = [], [], []
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

        # Validation Step
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for videos, labels in val_dl:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Test Step
        test_correct, test_total, test_loss = 0, 0, 0
        with torch.no_grad():
            for videos, labels in test_dl:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                preds = outputs.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_acc = 100 * test_correct / test_total
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
    if best_model_state_dict is not None:
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        best_model_path = os.path.join(MODEL_DIR, f"lrcn_best_{timestamp}.pth")
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(best_model_state_dict, best_model_path)
        logging.info(f"Best model saved at {best_model_path}")

    # Load best model for final evaluation
    model.load_state_dict(best_model_state_dict)
    model.eval()

    # Evaluate performance on all datasets
    train_final_acc = evaluate_performance(model, train_dl, "Train")
    val_final_acc = evaluate_performance(model, val_dl, "Validation")
    test_final_acc = evaluate_performance(model, test_dl, "Test")

    # Plot metrics (accuracy, loss)
    plot_metrics(train_accuracies, test_accuracies, val_accuracies, train_losses, test_losses, val_losses)

# Call the training and evaluation function
if __name__ == "__main__":
    train_and_evaluate()
