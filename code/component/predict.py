import torch
import cv2
import numpy as np
from model import LRCN
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
CLASSES_LIST = os.getenv("CLASSES_LIST")
IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH = os.getenv("IMAGE_HEIGHT"), os.getenv("IMAGE_WIDTH"), os.getenv("SEQUENCE_LENGTH")

def predict_video(video_path, model_path):
    """
    Predicts the class label for a given video using a pre-trained LRCN model.

    Args:
        video_path (str): The file path to the video that needs to be classified.
        model_path (str): The file path to the pre-trained model's state dictionary.

    Steps:
        1. Reads and extracts frames from the video at the specified `video_path`.
        2. Preprocesses the frames (resizing and normalizing).
        3. Loads the pre-trained LRCN model from the `model_path`.
        4. Passes the video frames through the model to obtain predictions.
        5. Returns the predicted class label based on the model's output.

    Returns:
        str: The predicted class label for the video, printed to the console.
        
    If the number of frames in the video is less than the required `SEQUENCE_LENGTH`, 
    the function will print "Not enough frames" and exit.

    Example:
        predict_video("path/to/video.mp4", "saved_models/lrcn_best.pth")
    """
    # Capture video frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(int(total / SEQUENCE_LENGTH), 1)
    for i in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        frame = frame / 255.0  # Normalize the frame
        frames.append(frame)
    cap.release()

    # If there are not enough frames, return early
    if len(frames) < SEQUENCE_LENGTH:
        print("Not enough frames")
        return

    # Convert frames into a tensor
    video = torch.FloatTensor(np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(0)

    # Load model and make prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LRCN(num_classes=len(CLASSES_LIST)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output = model(video.to(device))
        pred = output.argmax(dim=1).item()
    
    # Output predicted class
    return f"Predicted Class: {CLASSES_LIST[pred]}"
