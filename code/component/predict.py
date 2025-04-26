import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from model import LRCN
import os
from dotenv import load_dotenv
import json
# Load environment variables from .env file
load_dotenv()

def predict_video_class(video_path, model_path, class_names = json.loads(os.getenv("CLASSES_LIST")), sequence_length=int(os.getenv("SEQUENCE_LENGTH")), image_height=int(os.getenv("IMAGE_HEIGHT")), image_width=int(os.getenv("IMAGE_WIDTH")), device=None):
    """
    Predicts the class of a video using a trained LRCN model.

    Args:
        video_path (str): Path to the input video file.
        model_path (str): Path to the trained model weights (.pt file).
        class_names (list): List of class labels (index to string).
        sequence_length (int): Number of frames to sample from the video.
        image_height (int): Height of each frame.
        image_width (int): Width of each frame.
        device (str or torch.device): "cuda" or "cpu".

    Returns:
        str: Predicted class label.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the same transform used in training
    transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor()
    ])

    # Load the model and weights
    model = LRCN(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Extract frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(int(total_frames / sequence_length), 1)
    
    frames = []
    for i in range(sequence_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_width, image_height))
        frame_pil = Image.fromarray(frame)
        frame_tensor = transform(frame_pil)
        frames.append(frame_tensor)
    cap.release()

    if len(frames) < sequence_length:
        raise ValueError(f"Could only extract {len(frames)} frames, expected {sequence_length}")

    video_tensor = torch.stack(frames).unsqueeze(0).to(device)  # shape: [1, T, C, H, W]

    with torch.no_grad():
        output = model(video_tensor)
        predicted_idx = output.argmax(dim=1).item()
    
    return class_names[predicted_idx]

# video_path = "/home/ubuntu/Human-Activity-Recognition/Data/UCF50/BaseballPitch/v_BaseballPitch_g01_c01.avi"
# model_path = "/home/ubuntu/Human-Activity-Recognition/code/component/models/lrcn_best_2025_04_25__19_49_33.pth"
# predicted_class = predict_video_class(video_path, model_path)
# print("Predicted Class:", predicted_class)