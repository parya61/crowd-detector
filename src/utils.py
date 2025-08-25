import cv2
import os
from typing import List, Tuple

def draw_detections(frames: List, detections: List[List[Tuple[int, int, int, int, float]]]) -> List:
    """
    Draw bounding boxes and labels on each frame.
    """
    rendered_frames = []

    for frame, dets in zip(frames, detections):
        for (x1, y1, x2, y2, conf) in dets:
            label = f"person: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        rendered_frames.append(frame)

    return rendered_frames

def save_video(frames: List, output_path: str, fps: int = 25):
    """
    Save a sequence of frames to a video file.
    """
    if not frames:
        raise ValueError("No frames to save.")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"[INFO] Saved output video to {output_path}")