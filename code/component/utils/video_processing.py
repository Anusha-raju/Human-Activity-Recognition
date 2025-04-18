import cv2
import numpy as np

def extract_frames(video_path, sequence_length=20, width=64, height=64):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(int(total / sequence_length), 1)
    for i in range(sequence_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (width, height))
        frame = frame / 255.0
        frames.append(frame)
    cap.release()
    if len(frames) == sequence_length:
        return np.stack(frames)
    return None
