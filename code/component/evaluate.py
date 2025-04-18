import torch
from torch.utils.data import DataLoader
from model import LRCN
from utils.dataset import VideoDataset

DATASET_DIR = "data/UCF50"
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
BATCH_SIZE = 4

dataset = VideoDataset(DATASET_DIR, CLASSES_LIST)
_, val_ds = torch.utils.data.random_split(dataset, [int(0.75 * len(dataset)), len(dataset) - int(0.75 * len(dataset))])
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LRCN(num_classes=len(CLASSES_LIST)).to(device)
model.load_state_dict(torch.load("saved_models/lrcn_model_latest.pth"))
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for videos, labels in val_dl:
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"Validation Accuracy: {100 * correct / total:.2f}%")
