import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, data_dir, classes, sequence_length=20, image_height=64, image_width=64):
        self.data_dir = data_dir
        self.classes = classes
        self.sequence_length = sequence_length
        self.image_height = image_height
        self.image_width = image_width
        self.videos = []
        self.labels = []
        self._load_dataset()

    def _load_dataset(self):
        for class_index, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            for video_name in os.listdir(class_dir):
                path = os.path.join(class_dir, video_name)
                frames = self._extract_frames(path)
                if frames is not None:
                    self.videos.append(frames)
                    self.labels.append(class_index)

    def _extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(int(total / self.sequence_length), 1)
        for i in range(self.sequence_length):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            success, frame = cap.read()
            if not success:
                cap.release()
                return None
            frame = cv2.resize(frame, (self.image_width, self.image_height))
            frame = frame / 255.0
            frames.append(frame)
        cap.release()
        return np.stack(frames)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = torch.FloatTensor(self.videos[idx]).permute(0, 3, 1, 2)
        label = torch.tensor(self.labels[idx])
        return video, label
