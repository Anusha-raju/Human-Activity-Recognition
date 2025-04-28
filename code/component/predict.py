import torch
import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code/')))
from component.model import LRCN
from dotenv import load_dotenv
import json
from pathlib import Path

# Load environment variables from .env file
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
import requests

def download_model_from_s3():
    local_path = os.getenv("LOCAL_MODEL_PATH")
    s3_key = os.getenv("MODEL_PATH")
    response = requests.get(s3_key)
    with open(local_path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded model to {local_path}")

def find_model_path():
    current_file = Path(__file__).resolve()

    # Try two levels up
    two_up_dir = current_file.parents[2] / "code" / "maincode" / "models"
    if two_up_dir.exists() and any(two_up_dir.iterdir()):
        model_files = sorted(two_up_dir.iterdir(), key=lambda x: x.stat().st_mtime)
        return model_files[-1]

    # If not found, try one level up
    one_up_dir = current_file.parents[1] / "code" / "maincode" / "models"
    if one_up_dir.exists() and any(one_up_dir.iterdir()):
        model_files = sorted(two_up_dir.iterdir(), key=lambda x: x.stat().st_mtime)
        return model_files[-1]
        # first_file = next(one_up_dir.iterdir())
        # return first_file

    # If neither found
    raise FileNotFoundError("No model file found in 'code/maincode/models' folder.")

def predict_video_class(video_path, model_path = MODEL_PATH, class_names = json.loads(os.getenv("CLASSES_LIST")), sequence_length=int(os.getenv("SEQUENCE_LENGTH")), image_height=int(os.getenv("IMAGE_HEIGHT")), image_width=int(os.getenv("IMAGE_WIDTH")), device=None):
    """
    Classifies a video into one of the predefined categories using a trained LRCN (Long-term Recurrent Convolutional Network) model.

    Args:
        video_path (str): The file path to the input video.
        model_path (str): The path to the model's trained weights file (.pt).
        class_names (list): A list mapping index values to class labels.
        sequence_length (int): The number of frames to sample from the video.
        image_height (int): The height of each frame after resizing.
        image_width (int): The width of each frame after resizing.
        device (str or torch.device): The device to run the model on, either "cuda" for GPU or "cpu".

    Returns:
        str: The label of the predicted class for the video.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply the same image transformation used during model training
    transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor()
    ])

    # Load the trained LRCN model and weights
    model = LRCN(num_classes=len(class_names))
    if not model_path:
        model_path = find_model_path()
    else:
        download_model_from_s3()
        model_path = os.getenv("LOCAL_MODEL_PATH")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Open the video file and extract frames
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

    # Ensure enough frames were extracted
    if len(frames) < sequence_length:
        raise ValueError(f"Only {len(frames)} frames were extracted, but {sequence_length} were expected")

    # Stack frames into a tensor and move to the specified device
    video_tensor = torch.stack(frames).unsqueeze(0).to(device)  # shape: [1, T, C, H, W]

    with torch.no_grad():
        output = model(video_tensor)
        predicted_idx = output.argmax(dim=1).item()
    
    # Return the predicted class label
    return class_names[predicted_idx]

## Example
if __name__ == "__main__":
    pass
#     video_path = "/home/ubuntu/Human-Activity-Recognition/Data/UCF50/BaseballPitch/v_BaseballPitch_g01_c01.avi"
#     predicted_class = predict_video_class(video_path)
#     print("Predicted Class:", predicted_class)
