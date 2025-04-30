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
import requests
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code/')))
import boto3
from pathlib import Path
from component.model import LRCN
from component.utils.visualization import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np

# MODEL_DIR TO SAVED MODEL
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_PATH = os.getenv("MODEL_PATH")

# Logging setup using logging module
logging.basicConfig(filename=os.getenv("LOG_FILE_PATH"),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_model_from_s3():
    local_path = os.getenv("LOCAL_MODEL_PATH")
    # bucket_name = os.getenv("BUCKET_NAME")
    s3_key = os.getenv("MODEL_PATH")
    # s3 = boto3.client('s3')
    # Path(local_path).parent.mkdir(parents=True, exist_ok=True)  # create dir if not exists
    # s3.download_file(bucket_name, s3_key, local_path)
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
    three_up_dir = current_file.parents[3] / "code" / "maincode" / "models"
    if three_up_dir.exists() and any(three_up_dir.iterdir()):
        model_files = sorted(three_up_dir.iterdir(), key=lambda x: x.stat().st_mtime)
        return model_files[-1]

    # If neither found
    raise FileNotFoundError("No model file found in 'code/maincode/models' folder.")


def evaluate_model(model, data_loader, criterion, data_type="Test", confusion_matrix_ = False):
    """
    Evaluates the performance of the model on a specific dataset (train/validation/test).
    
    Args:
        model (nn.Module): The model that is being evaluated.
        data_loader (DataLoader): A DataLoader object that loads the dataset.
        criterion (torch.nn.Module): The loss function used to compute the model's error.
        data_type (str, optional): Specifies the type of dataset (e.g., "Test", "Train", "Validation").
    
    Returns:
        accuracy (float): The percentage accuracy of the model on the dataset.
        loss (float): The average loss computed over the dataset.
    """
    y_true = []
    y_pred = []
    class_names = json.loads(os.getenv("CLASSES_LIST"))
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for videos, labels in data_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    if confusion_matrix_:
    # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        mcm = multilabel_confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        total_cm = mcm.sum(axis=0)
        tn, fp, fn, tp = total_cm.ravel()

        logging.info(f"Overall True Positives: {tp}")
        logging.info(f"Overall True Negatives: {tn}")
        logging.info(f"Overall False Positives: {fp}")
        logging.info(f"Overall False Negatives: {fn}")
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plot_confusion_matrix(cm_normalized , class_names)
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    return accuracy, avg_loss

def save_best_model(model, model_state_dict):
    """
    Saves the best model based on its test accuracy.

    Args:
        model (nn.Module): The model to be saved.
        model_state_dict (dict): The model's state dictionary containing the weights.
    
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
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be extracted.
    
    Returns:
        float: The current learning rate from the first parameter group of the optimizer.

    Example:
        current_lr = get_current_lr(optimizer)
        print(f"Current learning rate: {current_lr}")
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def load_model():
    """
    Loads a pre-trained model from disk.

    Args:
    Returns:
        model: The pre-trained LRCN model.
    """
    model = LRCN(num_classes=len(json.loads(os.getenv("CLASSES_LIST"))))
    model_path = MODEL_PATH
    if not model_path:
        model_path = find_model_path()
    else:
        download_model_from_s3()
        model_path = os.getenv("LOCAL_MODEL_PATH")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def process_video_frames(video_path):
    """
    Processes video frames, converting them to tensors for model input.

    Args:
        video_path (str): The path to the input video file.
    
    Returns:
        frames (list): A list of original frames from the video.
        transformed (list): A list of transformed frames, each represented as a tensor.
    """
    image_height = int(os.getenv("IMAGE_HEIGHT"))
    image_width = int(os.getenv("IMAGE_WIDTH"))

    sequence_length = int(os.getenv("SEQUENCE_LENGTH"))
    # Define the transformations to apply to each frame (resize and to tensor)
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
        transformed.append(tensor)  # Transformed frame (used for model prediction)
    
    cap.release()
    return frames, transformed
