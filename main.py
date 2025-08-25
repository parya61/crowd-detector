import argparse
from src.detector import CrowdDetector
from src.utils import draw_detections, save_video


def main(video_path: str, output_path: str):
    model = CrowdDetector()
    frames, detections = model.process_video(video_path)
    rendered_frames = draw_detections(frames, detections)
    save_video(rendered_frames, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect people in video and draw bounding boxes.")
    parser.add_argument("--input", required=True, help="Path to input video file (e.g., crowd.mp4)")
    parser.add_argument("--output", required=True, help="Path to save output video")
    args = parser.parse_args()

    main(args.input, args.output)