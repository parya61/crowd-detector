import cv2
import os
from typing import List, Tuple


class CrowdDetector:
    def __init__(self):
        model_dir = os.path.join(os.path.dirname(__file__), '../models')
        self.prototxt = os.path.join(model_dir, 'deploy.prototxt')
        self.weights = os.path.join(model_dir, 'mobilenet_iter_73000.caffemodel')

        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.weights)
        self.person_class_id = 15  # 'person' class in COCO dataset

    def process_video(self, video_path: str) -> Tuple[List, List]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        detections_per_frame = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detect_people(frame)
            print(f"Кадр: {len(frames)+1}, обнаружено людей:{len(detections)}")
            frames.append(frame)
            detections_per_frame.append(detections)

        cap.release()
        return frames, detections_per_frame

    def detect_people(self, frame) -> List[Tuple[int, int, int, int, float]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        output = self.net.forward()

        people = []
        for i in range(output.shape[2]):
            confidence = output[0, 0, i, 2]
            class_id = int(output[0, 0, i, 1])

            if class_id == self.person_class_id and confidence > 0.5:
                box = output[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")
                people.append((x1, y1, x2, y2, confidence))

        return people