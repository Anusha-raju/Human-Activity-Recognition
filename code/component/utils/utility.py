import logging
import torch
import datetime
import os
from dotenv import load_dotenv
load_dotenv()
# MODEL_DIR TO SAVED MODEL
MODEL_DIR = os.getenv("MODEL_DIR", "models")
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
