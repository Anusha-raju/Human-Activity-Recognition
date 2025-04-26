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
    Extracts CNN features from the model.

    Args:
        model (nn.Module): The LRCN model.
        input_tensor (torch.Tensor): The input tensor.
        cnn_features (list): List to store the extracted features.

    Returns:
        None: The features are stored in the `cnn_features` list.
    """
    def hook_fn(module, input, output):
        cnn_features.append(output.detach().cpu())  # Store feature maps

    hook = model.feature_extractor.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_tensor)  # Forward pass to trigger the hook
    hook.remove()

def generate_heatmap(frames, cnn_features, sequence_length):
    """
    Generates a heatmap and overlays it on the original frames.

    Args:
        frames (list): List of original frames.
        cnn_features (list): List of CNN features.
        sequence_length (int): Number of frames.

    Returns:
        overlays (list): List of frames with heatmap overlays.
    """
    overlays = []
    for i in range(sequence_length):
        fmap = cnn_features[0][i].mean(0).numpy()  # Get the average feature map for each frame
        fmap = cv2.resize(fmap, (frames[i].shape[1], frames[i].shape[0]))  # Resize to match frame size
        heatmap = cv2.applyColorMap(np.uint8(255 * fmap / fmap.max()), cv2.COLORMAP_JET)  # Convert to heatmap
        overlay = cv2.addWeighted(frames[i], 0.6, heatmap, 0.4, 0)  # Overlay heatmap on original frame
        overlays.append(overlay)
    return overlays

def generate_activity_labels(frames, model, transformed, sequence_length, class_names, device):
    """
    Generate activity labels for each frame.

    Args:
        frames (list): List of original frames.
        model (nn.Module): The LRCN model.
        transformed (list): List of transformed frames.
        sequence_length (int): Number of frames.
        class_names (list): List of class labels.
        device (torch.device): The device (CPU or GPU).

    Returns:
        labeled_frames (list): List of frames with activity labels.
        labels (list): List of predicted activity labels.
    """
    labeled_frames = []
    labels = []

    input_tensor = torch.stack(transformed).unsqueeze(0).to(device)  # [1, T, C, H, W]
    with torch.no_grad():
        output = model(input_tensor)  # Predict activity for the whole sequence
    
    # Generate per-frame activity labels
    for i in range(sequence_length):
        predicted_idx = output[0][i].argmax().item()  # Get prediction for each frame
        predicted_label = class_names[predicted_idx]
        labels.append(predicted_label)
        
        # Overlay the predicted label on the frame
        label_text = f"Predicted: {predicted_label}"
        frame = frames[i]
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        labeled_frames.append(frame)

    return labeled_frames, labels


def save_output_video(overlays, output_path, fps):
    """
    Saves the final output video with overlays.

    Args:
        overlays (list): List of frames with overlays.
        output_path (str): Path to save the output video.
        fps (float): Frames per second of the video.

    Returns:
        None: Saves the video to `output_path`.
    """
    h, w, _ = overlays[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    for frame in overlays:
        out.write(frame)
    out.release()

def video_analysis(video_path, option="heatmap"):
    """
    Generates a video with either CNN feature heatmaps or activity labels per frame.

    Args:
        video_path (str): Path to the input video.
        model_path (str): Path to the trained LRCN model weights.
        output_filename (str): Name of the output file.
        option (str): The option for analysis ('heatmap' or 'option2').

    Returns:
        str: Path to the output video file.
    """
    analysis_dir = os.getenv("ANALYSIS")
    os.makedirs(analysis_dir, exist_ok=True)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    # Load model
    model = load_model()
    # Process video frames
    sequence_length = int(os.getenv("SEQUENCE_LENGTH"))
    frames, transformed = process_video_frames(video_path)

    if len(transformed) < sequence_length:
        print(f"Only got {len(transformed)} frames, expected {sequence_length}")
        return

    # Stack the transformed frames and send to model
    input_tensor = torch.stack(transformed).unsqueeze(0).to(device)  # [1, T, C, H, W]
    
    # Extract CNN features
    cnn_features = []
    extract_cnn_features(model, input_tensor, cnn_features)

    if option == "heatmap":
        # Generate heatmap overlays
        cnn_features = []
        extract_cnn_features(model, torch.stack(transformed).unsqueeze(0).to(device), cnn_features)
        overlays = generate_heatmap(frames, cnn_features, sequence_length)
        output_path = os.path.join(analysis_dir, f"{video_basename}_heatmap.avi")

    elif option == "activity_labels":
        # Generate activity labels per frame
        class_names = json.loads(os.getenv("CLASSES_LIST"))
        labeled_frames, labels = generate_activity_labels(frames, model, transformed, sequence_length, class_names, device)
        overlays = labeled_frames # For now, just return the original frames as placeholder
        output_path = os.path.join(analysis_dir, f"{video_basename}_activity_labels.avi")


    # Save the output video
    save_output_video(overlays, output_path, cv2.CAP_PROP_FPS)

    absolute_output_path = os.path.abspath(output_path)
    print(f"Analysis video saved to: {absolute_output_path}")
    return absolute_output_path

if __name__ == "__main__":
    # Example of calling the function with the "heatmap" option
    # video_path = "/path/to/video/file.avi"
    # video_heatmap(video_path, option="heatmap")
    pass