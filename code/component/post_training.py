import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code/')))
from component.model import LRCN
from component.utils.utility import load_model, process_video_frames
import json
load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_cnn_features(model, input_tensor, cnn_features):
    """
    Extracts the feature maps from the model's CNN layers and appends them to a list.

    Args:
        model (nn.Module): The LRCN model to extract features from.
        input_tensor (torch.Tensor): The input tensor containing video frame data.
        cnn_features (list): A list where the extracted CNN features will be stored.

    Returns:
        None: The features are added to the `cnn_features` list.
    """
    def hook_fn(module, input, output):
        cnn_features.append(output.detach().cpu())  # Store the feature maps

    hook = model.feature_extractor.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_tensor)  # Perform a forward pass to capture the features
    hook.remove()

def generate_heatmap(frames, cnn_features, sequence_length):
    """
    Generates heatmaps based on CNN features and overlays them onto the original video frames.

    Args:
        frames (list): List of original video frames.
        cnn_features (list): List of CNN feature maps for each frame.
        sequence_length (int): The total number of frames in the sequence.

    Returns:
        overlays (list): List of frames with heatmap overlays applied.
    """
    overlays = []
    for i in range(sequence_length):
        fmap = cnn_features[0][i].mean(0).numpy()  # Average the feature map across channels
        fmap = cv2.resize(fmap, (frames[i].shape[1], frames[i].shape[0]))  # Resize to match frame dimensions
        heatmap = cv2.applyColorMap(np.uint8(255 * fmap / fmap.max()), cv2.COLORMAP_JET)  # Apply color map to the feature map
        overlay = cv2.addWeighted(frames[i], 0.6, heatmap, 0.4, 0)  # Overlay the heatmap on the original frame
        overlays.append(overlay)
    return overlays

def generate_activity_labels(frames, model, transformed, sequence_length, class_names, device):
    """
    Generates predicted activity labels for each frame in the video sequence.

    Args:
        frames (list): List of original video frames.
        model (nn.Module): The LRCN model used for prediction.
        transformed (list): List of transformed video frames ready for input to the model.
        sequence_length (int): The number of frames in the sequence.
        class_names (list): List of possible activity class labels.
        device (torch.device): The device (CPU or GPU) for model inference.

    Returns:
        labeled_frames (list): List of frames with predicted activity labels overlaid.
        labels (list): List of predicted activity labels for each frame.
    """
    labeled_frames = []
    labels = []

    input_tensor = torch.stack(transformed).unsqueeze(0).to(device)  # Create input tensor
    with torch.no_grad():
        output = model(input_tensor)  # Run model inference for the sequence
    
    # Generate activity labels for each frame
    for i in range(sequence_length):
        predicted_idx = output[0][i].argmax().item()  # Get the predicted class index for each frame
        predicted_label = class_names[predicted_idx]  # Map index to class label
        labels.append(predicted_label)
        
        # Annotate the frame with the predicted label
        label_text = f"Predicted: {predicted_label}"
        frame = frames[i]
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        labeled_frames.append(frame)

    return labeled_frames, labels

def save_output_video(overlays, output_path, fps):
    """
    Saves the processed video (with overlays) to a specified file path.

    Args:
        overlays (list): List of video frames with overlays applied.
        output_path (str): Path where the output video will be saved.
        fps (float): Frames per second of the video.

    Returns:
        None: Saves the output video to the specified location.
    """
    h, w, _ = overlays[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    for frame in overlays:
        out.write(frame)  # Write each frame to the output video
    out.release()

def video_analysis(video_path, option="heatmap"):
    """
    Analyzes the input video and generates either heatmap overlays or activity labels.

    Args:
        video_path (str): Path to the input video file.
        option (str): Type of analysis to perform ('heatmap' or 'activity_labels').

    Returns:
        str: The absolute file path to the saved output video.
    """
    analysis_dir = os.getenv("ANALYSIS")
    os.makedirs(analysis_dir, exist_ok=True)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    # Load the model
    model = load_model()
    # Process the video frames
    sequence_length = int(os.getenv("SEQUENCE_LENGTH"))
    frames, transformed = process_video_frames(video_path)

    if len(transformed) < sequence_length:
        print(f"Only got {len(transformed)} frames, expected {sequence_length}")
        return

    # Stack the transformed frames and send them to the model
    input_tensor = torch.stack(transformed).unsqueeze(0).to(device)
    
    # Extract CNN features
    cnn_features = []
    extract_cnn_features(model, input_tensor, cnn_features)

    if option == "heatmap":
        # Generate and overlay heatmaps
        cnn_features = []
        extract_cnn_features(model, torch.stack(transformed).unsqueeze(0).to(device), cnn_features)
        overlays = generate_heatmap(frames, cnn_features, sequence_length)
        output_path = os.path.join(analysis_dir, f"{video_basename}_heatmap.avi")

    elif option == "activity_labels":
        # Generate activity labels for each frame
        class_names = json.loads(os.getenv("CLASSES_LIST"))
        labeled_frames, labels = generate_activity_labels(frames, model, transformed, sequence_length, class_names, device)
        overlays = labeled_frames  # Use the frames with labels as the overlays
        output_path = os.path.join(analysis_dir, f"{video_basename}_activity_labels.avi")

    # Save the resulting video
    save_output_video(overlays, output_path, cv2.CAP_PROP_FPS)

    absolute_output_path = os.path.abspath(output_path)
    print(f"Analysis video saved to: {absolute_output_path}")
    return absolute_output_path

if __name__ == "__main__":
    # Example of calling the function with the "heatmap" option
    # video_path = "/path/to/video/file.avi"
    # video_heatmap(video_path, option="heatmap")
    pass