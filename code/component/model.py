import torch
import torch.nn as nn
import os
from dotenv import load_dotenv
load_dotenv()

dropout_prob = float(os.getenv("DROPOUT"))
class LRCN(nn.Module):
    def __init__(self, num_classes):
        super(LRCN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.Dropout2d(p=dropout_prob),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.Dropout2d(p=dropout_prob),
            nn.MaxPool2d(2),
        )
        self.lstm = nn.LSTM(input_size=128 * 16 * 16, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.feature_extractor(x)
        x = x.reshape(b, t, -1)
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        return x