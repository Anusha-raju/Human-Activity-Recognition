import torch
import cv2
import numpy as np
from model import LRCN

CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH = 64, 64, 20

def predict_video(video_path, model_path):
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
        frame = frame / 255.0
        frames.append(frame)
    cap.release()

    if len(frames) < SEQUENCE_LENGTH:
        print("Not enough frames")
        return

    video = torch.FloatTensor(np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LRCN(num_classes=len(CLASSES_LIST)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output = model(video.to(device))
        pred = output.argmax(dim=1).item()
    print(f"Predicted Class: {CLASSES_LIST[pred]}")
