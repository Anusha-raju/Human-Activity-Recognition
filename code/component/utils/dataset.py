import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

project_root = Path(__file__).resolve().parents[3]

class VideoDataset(Dataset):
    """
    Custom dataset for loading and processing videos for action recognition.
    Each video is split into frames, and the dataset returns sequences of frames (as tensors) and their corresponding labels.

    Args:
        data_dir (str): Path to the root directory containing subdirectories for each class.
        classes (list of str): List of class names corresponding to video categories.
        sequence_length (int, optional): Number of frames per video sequence (default is 20).
        image_height (int, optional): Height to which each video frame will be resized (default is 64).
        image_width (int, optional): Width to which each video frame will be resized (default is 64).
        transform (callable, optional): A function/transform to apply to each frame (default is a composed transformation for augmentation).
    """
    def __init__(self, data_dir, classes, sequence_length=int(os.getenv("SEQUENCE_LENGTH")), image_height=int(os.getenv("IMAGE_HEIGHT")), image_width=int(os.getenv("IMAGE_WIDTH"))):
        """
        Initializes the dataset by loading video paths and corresponding labels, and applies any given transformations.

        Args:
            data_dir (str): Path to the root directory containing subdirectories for each class.
            classes (list of str): List of class names corresponding to video categories.
            sequence_length (int): Number of frames to sample from each video.
            image_height (int): Height to which each video frame will be resized.
            image_width (int): Width to which each video frame will be resized.
            transform (callable, optional): A function/transform to apply to each frame (default is a composed transformation for augmentation).
        """
        self.data_dir = os.path.join(project_root, data_dir)
        self.classes = classes
        self.sequence_length = sequence_length
        self.image_height = image_height
        self.image_width = image_width
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(size=(self.image_height, self.image_width), scale=(0.8, 1.0)),
            transforms.GaussianBlur(kernel_size=5),
            transforms.RandomGrayscale(p=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
        ])
        self.videos = []  # List to store videos as sequences of frames
        self.labels = []  # List to store corresponding class labels
        self._load_dataset()

    def _load_dataset(self):
        """
        Loads the dataset by iterating over each class directory, extracting frames from each video,
        and storing the frames and corresponding labels.

        This function populates `self.videos` with video frame sequences and `self.labels` with corresponding
        class indices (labels).

        Iterates over:
            - Each class directory.
            - Each video in the class directory.
            - Extracts frames for each video.
        """
        for class_index, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            for video_name in os.listdir(class_dir):
                path = os.path.join(class_dir, video_name)
                frames = self._extract_frames(path)  # Extract frames from video
                if frames is not None:
                    self.videos.append(frames)  # Store video frames
                    self.labels.append(class_index)  # Store corresponding label (class index)

    def _extract_frames(self, video_path):
        """
        Extracts a fixed number of frames from a video file, resizing them to the specified dimensions.

        Args:
            video_path (str): Path to the video file.

        Returns:
            list of np.ndarray: A list containing the frames as numpy arrays if extraction is successful, 
                                 None if the video could not be processed.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
        step = max(int(total / self.sequence_length), 1)  # Calculate frame step to get 'sequence_length' frames

        # Extract frames at regular intervals
        for i in range(self.sequence_length):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)  # Set the current frame position
            success, frame = cap.read()  # Read the frame
            if not success:
                cap.release()
                return None  # If reading fails, return None
            frame = cv2.resize(frame, (self.image_width, self.image_height))  # Resize frame
            frames.append(frame)  # Append frame to list
        cap.release()  # Release video capture object
        return frames

    def __len__(self):
        """
        Returns the total number of video sequences in the dataset.

        Returns:
            int: Number of video sequences in the dataset.
        """
        return len(self.videos)

    def __getitem__(self, idx):
        """
        Retrieves a single video sequence and its corresponding label from the dataset at the specified index.
        Applies any transformations to the video frames if provided.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (video_tensor, label_tensor)
                - video_tensor (torch.Tensor): Tensor of shape [T, C, H, W] representing the video sequence, 
                  where T is the number of frames, C is the number of channels (3 for RGB), H is the height, 
                  and W is the width of each frame.
                - label_tensor (torch.Tensor): Tensor of the corresponding label (class index).
        """
        video = self.videos[idx]  # Get the video sequence
        label = self.labels[idx]  # Get the corresponding label

        transformed_frames = []
        for frame in video:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame from BGR (OpenCV) to RGB
            frame_pil = Image.fromarray(frame)  # Convert NumPy array to PIL image

            if self.transform:
                frame_pil = self.transform(frame_pil)  # Apply the transformation

            transformed_frames.append(frame_pil)  # Store transformed frame

        # Stack the frames to create a tensor of shape [T, C, H, W]
        video_tensor = torch.stack(transformed_frames)
        label_tensor = torch.tensor(label)  # Convert the label to a tensor
        return video_tensor, label_tensor
