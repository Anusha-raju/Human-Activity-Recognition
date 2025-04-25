import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
load_dotenv()

# Load output path for plots from environment variable
plot_output_path = os.getenv("PLOT_OUTPUT_PATH")

# Ensure the output directory exists
if not os.path.exists(plot_output_path):
    os.makedirs(plot_output_path)

def plot_metrics(train_acc, test_acc, val_acc, train_loss, test_loss, val_loss):
    """
    Plots and saves accuracy and loss curves for training, testing, and validation metrics.

    Args:
        train_acc (list): A list of training accuracy values across epochs.
        test_acc (list): A list of testing accuracy values across epochs.
        val_acc (list): A list of validation accuracy values across epochs.
        train_loss (list): A list of training loss values across epochs.
        test_loss (list): A list of testing loss values across epochs.
        val_loss (list): A list of validation loss values across epochs.

    Saves two plots:
        1. Accuracy Curve: A plot showing the accuracy for train, test, and validation sets over epochs.
        2. Loss Curve: A plot showing the loss for train, test, and validation sets over epochs.
        
    The plots are saved as PNG files at the directory specified by the `PLOT_OUTPUT_PATH` environment variable.

    Returns:
        None
    """
    # Plot accuracy curves
    plt.figure()
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')

    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Save the accuracy curve plot to the output path
    plt.savefig(os.path.join(plot_output_path, 'accuracy_curve.png'))
    plt.close()

    # Plot loss curves
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.plot(val_loss, label='Validation Loss')

    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the loss curve plot to the output path
    plt.savefig(os.path.join(plot_output_path, 'loss_curve.png'))
    plt.close()
