import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import LRCN
from utils.dataset import VideoDataset
from utils.visualization import plot_metrics
import datetime
import os
import logging

# Set seed for reproducibility
SEED = 27
torch.manual_seed(SEED)

# Config
DATASET_DIR = "Data/UCF50"
CLASSES_LIST = [
    "BaseballPitch", "Basketball", "BenchPress", "Biking", "Billiards", "BreastStroke",
    "CleanAndJerk", "Diving", "Drumming", "Fencing", "GolfSwing", "HighJump",
    "HorseRace", "HorseRiding", "HulaHoop", "JavelinThrow", "JugglingBalls",
    "JumpRope", "JumpingJack", "Kayaking", "Lunges", "MilitaryParade", "Mixing",
    "Nunchucks", "PizzaTossing", "PlayingGuitar", "PlayingPiano", "PlayingTabla",
    "PlayingViolin", "PoleVault", "PommelHorse", "PullUps", "Punch", "PushUps",
    "RockClimbingIndoor", "RopeClimbing", "Rowing", "SalsaSpin", "SkateBoarding",
    "Skiing", "Skijet", "SoccerJuggling", "Swing", "TaiChi", "TennisSwing",
    "ThrowDiscus", "TrampolineJumping", "VolleyballSpiking", "WalkingWithDog", "YoYo"
]
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4

# Dataset and Dataloaders
dataset = VideoDataset(DATASET_DIR, CLASSES_LIST)
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LRCN(num_classes=len(CLASSES_LIST)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_test_acc = 0
best_model_path = None

# Logging setup using logging module
log_file = "train.log"

logging.basicConfig(filename=log_file,
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Epoch, Train Acc, Val Acc, Test Acc")

# Training loop
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
    acc = 100 * correct / total
    train_losses.append(total_loss)
    train_accuracies.append(acc)

    # Validation
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

    # Test evaluation
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for videos, labels in test_dl:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    test_acc = 100 * test_correct / test_total

    # Save best model
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        best_model_path = f"saved_models/lrcn_best_{timestamp}.pth"
        os.makedirs("saved_models", exist_ok=True)
        torch.save(model.state_dict(), best_model_path)

    # Log metrics to train.log
    logging.info(f"{epoch+1}, {acc:.2f}, {val_acc:.2f}, {test_acc:.2f}")

    print(f"[Epoch {epoch+1}] Train Acc: {acc:.2f}%, Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%")

# Load best model for final evaluation
model.load_state_dict(torch.load(best_model_path))
model.eval()

def evaluate(model, dataloader, name=""):
    correct, total = 0, 0
    with torch.no_grad():
        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"{name} Accuracy: {acc:.2f}%")
    return acc

# Final evaluation on all datasets
train_final_acc = evaluate(model, train_dl, "Train")
val_final_acc = evaluate(model, val_dl, "Validation")
test_final_acc = evaluate(model, test_dl, "Test")

# Plot metrics
plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses)
