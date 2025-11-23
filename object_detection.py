import argparse
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO

#!/usr/bin/env python3
"""
Basic object detection using OpenCV + Ultralytics YOLO11-nano.
Detects selected COCO classes from a USB camera and shows a live window.
Press ESC to exit.

It only detects the objects from the COCO classes defined in TARGET_CLASS_IDS.
For other classes, no boxes will be drawn.
For the fall detection application, use fall_detection.py instead.
"""

# Target COCO classes (names -> COCO class ids)
TARGET_CLASS_IDS = {
    39: "bottle",       # water bottle
    67: "cell phone",   # cellphone
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    24: "backpack",
}

def as_numpy(x):
    """Convert tensor/array to numpy array robustly."""
    try:
        return x.cpu().numpy()
    except Exception:
        return np.array(x)

def draw_box(frame, xyxy, label, conf, color=(0, 200, 0)):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def main():
    parser = argparse.ArgumentParser(description="YOLO11-nano USB camera demo (COCO subset).")
    parser.add_argument("--model", "-m", default="yolo11n.pt", help="Path or name of YOLO11-nano model (default: yolo11n.pt)")
    parser.add_argument("--device", "-d", default="cpu", help="Device for inference (cpu or cuda)")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device index (default 0)")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    # Load model (will download if model string points to a known remote)
    model = YOLO(args.model)
    # Set model confidence threshold (if supported) or pass per-call
    try:
        model.conf = args.conf
    except Exception:
        pass

    # Open camera
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2 if sys.platform.startswith("linux") else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    time.sleep(0.5)  # warm-up

    if not cap.isOpened():
        print("ERROR: Unable to open camera. Check camera index or permissions.", file=sys.stderr)
        sys.exit(1)

    window_name = "YOLO11-nano - press ESC to exit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Empty frame.", file=sys.stderr)
                break

            # Inference (model accepts BGR frames)
            results = model(frame, device=args.device, conf=args.conf, verbose=False)
            if len(results) == 0:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) == 27:
                    break
                continue
            r = results[0]

            # Extract boxes, classes, confidences robustly
            try:
                xyxy = as_numpy(r.boxes.xyxy)      # shape (N,4)
                cls_ids = as_numpy(r.boxes.cls).astype(int)
                confs = as_numpy(r.boxes.conf)
            except Exception:
                # If no boxes present
                xyxy = np.zeros((0, 4))
                cls_ids = np.zeros((0,), dtype=int)
                confs = np.zeros((0,))

            # Draw only target classes
            for bb, cid, cf in zip(xyxy, cls_ids, confs):
                if int(cid) in TARGET_CLASS_IDS:
                    label = TARGET_CLASS_IDS[int(cid)]
                    draw_box(frame, bb, label, float(cf))

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) == 27:  # ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()