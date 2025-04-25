import logging
import torch
import os
from dotenv import load_dotenv
load_dotenv()

# Logging setup using logging module
logging.basicConfig(filename=os.getenv("LOG_FILE_PATH"),
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def evaluate_performance(model, dataloader, name=""):
    """
    Evaluates the performance (accuracy) of a model on a given dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The DataLoader object containing the dataset to evaluate on.
        name (str, optional): A name for the dataset (e.g., "Train", "Test", "Validation"). Default is an empty string.

    Returns:
        float: The accuracy of the model on the provided dataset.

    Example:
        test_acc = evaluate_performance(model, test_dl, "Test")
        print(f"Test Accuracy: {test_acc:.2f}%")
    """
    correct, total = 0, 0
    with torch.no_grad():
        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    logging.info(f"{name} Accuracy: {acc:.2f}%")
    return acc