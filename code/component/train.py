import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import LRCN
from utils.dataset import VideoDataset
from utils.visualization import plot_metrics
import datetime

SEED = 27
torch.manual_seed(SEED)

# Config
DATASET_DIR = "data/UCF50"
CLASSES_LIST = [
    "BaseballPitch", "Basketball", "BenchPress", "Biking", "Billiards",
    "BreastStroke", "CleanAndJerk", "Diving", "Drumming", "Fencing",
    "GolfSwing", "PlayingGuitar", "HighJump", "HorseRace", "HorseRiding",
    "HulaHoop", "JavelinThrow", "JugglingBalls", "JumpRope", "JumpingJack",
    "Kayaking", "Lunges", "MilitaryParade", "Mixing", "Nunchucks",
    "PlayingPiano", "PizzaTossing", "PoleVault", "PommelHorse", "PullUps",
    "Punch", "PushUps", "RockClimbingIndoor", "RopeClimbing", "Rowing",
    "SalsaSpin", "SkateBoarding", "Skiing", "Skijet", "SoccerJuggling",
    "Swing", "Tabla", "TaiChi", "TennisSwing", "TrampolineJumping",
    "PlayingViolin", "VolleyballSpiking", "WalkingWithDog", "YoYo"
]
# CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4

dataset = VideoDataset(DATASET_DIR, CLASSES_LIST)
train_size = int(0.75 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LRCN(num_classes=len(CLASSES_LIST)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

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
    print(f"[Epoch {epoch+1}] Train Acc: {acc:.2f}%, Val Acc: {val_acc:.2f}%")

plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses)

timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
torch.save(model.state_dict(), f"saved_models/lrcn_model_{timestamp}.pth")
