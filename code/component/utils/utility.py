import logging
import torch
import datetime
import os
import cv2
from PIL import Image
import json
from torchvision import transforms
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code/')))

from component.model import LRCN

# MODEL_DIR TO SAVED MODEL
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.getenv("MODEL_PATH")
# Logging setup using logging module
logging.basicConfig(filename=os.getenv("LOG_FILE_PATH"),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, data_loader, criterion, data_type="Test"):
    """
    Evaluate the model on a given dataset (train/validation/test).
    
    Args:
        model (nn.Module): The trained model.
        data_loader (DataLoader): The DataLoader for the dataset.
        data_type (str): The type of the dataset (train/validation/test).
    
    Returns:
        accuracy (float): The accuracy of the model on the dataset.
        loss (float): The loss of the model on the dataset.
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for videos, labels in data_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    return accuracy, avg_loss

def save_best_model(model, model_state_dict):
    """
    Save the best model based on the test accuracy.
    
    Args:
        model (nn.Module): The model to save.
        model_state_dict (dict): The state dictionary of the best model.
    
    Returns:
        None
    """
    if model_state_dict is not None:
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        best_model_path = os.path.join(MODEL_DIR, f"lrcn_best_{timestamp}.pth")
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model_state_dict, best_model_path)
        logging.info(f"Best model saved at {best_model_path}")

def get_current_lr(optimizer):
    """
    Retrieves the current learning rate from the optimizer's parameter groups.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer object from which the learning rate will be extracted.

    Returns:
        float: The current learning rate from the optimizer's first parameter group.

    Example:
        current_lr = get_current_lr(optimizer)
        print(f"Current learning rate: {current_lr}")
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def load_model():
    """
    Loads the pre-trained model.

    Args:
    Returns:
        model: The loaded LRCN model.
    """
    model = LRCN(num_classes=len(json.loads(os.getenv("CLASSES_LIST"))))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def process_video_frames(video_path):
    """
    Processes video frames and converts them to tensors.

    Args:
        video_path (str): Path to the input video file.
        sequence_length (int): Number of frames to extract.
        transform (torchvision.transforms.Compose): The transformation to apply to each frame.

    Returns:
        frames (list): List of original frames.
        transformed (list): List of transformed frames (tensor format).
    """
    image_height=int(os.getenv("IMAGE_HEIGHT"))
    image_width=int(os.getenv("IMAGE_WIDTH"))

    sequence_length=int(os.getenv("SEQUENCE_LENGTH"))
    # Define the same transform used in training
    transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor()
    ])
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(int(total_frames / sequence_length), 1)

    frames = []
    transformed = []

    for i in range(sequence_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (image_width, image_height))
        pil_image = Image.fromarray(resized)
        tensor = transform(pil_image)
        frames.append(frame)  # Original frame
        transformed.append(tensor)  # Transformed frame (used for prediction)
    
    cap.release()
    return frames, transformed