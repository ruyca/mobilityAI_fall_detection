#!/usr/bin/env python3
"""
Enhanced object fall detection for Metro/Train rail monitoring.
Detects when objects fall into the rails danger zone using YOLO11 + motion tracking.
Press ESC to exit, 'r' to redefine danger zone.
"""

import argparse
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import requests

# Target COCO classes (names -> COCO class ids)
TARGET_CLASS_IDS = {
    39: "bottle",       # water bottle
    67: "cell phone",   # cellphone
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    24: "backpack",
}

@dataclass
class TrackedObject:
    """Represents a tracked object with motion history."""
    id: int
    class_name: str
    positions: deque  # Store last N positions [(x, y, timestamp)]
    velocities: deque  # Store last N velocities [(vx, vy)]
    last_bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    is_falling: bool = False
    fall_start_time: Optional[float] = None
    entered_danger_zone: bool = False

class ObjectTracker:
    """Simple centroid-based object tracker with motion analysis."""
    
    def __init__(self, max_disappeared=10, max_distance=50, history_size=10):
        self.next_object_id = 0
        self.objects: Dict[int, TrackedObject] = {}
        self.disappeared = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.history_size = history_size
        
        # Fall detection parameters
        self.fall_velocity_threshold = 15  # pixels/frame downward
        self.fall_acceleration_threshold = 2  # pixels/frame²
        self.min_fall_frames = 3  # minimum frames to confirm fall
        
    def register(self, centroid, bbox, class_name, confidence):
        """Register a new object."""
        obj = TrackedObject(
            id=self.next_object_id,
            class_name=class_name,
            positions=deque(maxlen=self.history_size),
            velocities=deque(maxlen=self.history_size-1),
            last_bbox=bbox,
            confidence=confidence
        )
        obj.positions.append((*centroid, time.time()))
        self.objects[self.next_object_id] = obj
        self.next_object_id += 1
        return obj.id
        
    def deregister(self, object_id):
        """Remove an object from tracking."""
        if object_id in self.objects:
            del self.objects[object_id]
            if object_id in self.disappeared:
                del self.disappeared[object_id]
                
    def update(self, detections):
        """Update tracker with new detections.
        detections: list of (bbox, class_name, confidence) tuples
        """
        # If no detections, increment disappeared count
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
            
        # Extract centroids from detections
        input_centroids = []
        for (bbox, _, _) in detections:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            input_centroids.append((cx, cy))
        input_centroids = np.array(input_centroids)
        
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for i, (bbox, class_name, conf) in enumerate(detections):
                self.register(input_centroids[i], bbox, class_name, conf)
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = []
            
            for obj_id in object_ids:
                obj = self.objects[obj_id]
                if len(obj.positions) > 0:
                    last_pos = obj.positions[-1]
                    object_centroids.append((last_pos[0], last_pos[1]))
            
            if len(object_centroids) > 0:
                object_centroids = np.array(object_centroids)
                
                # Compute distances between existing and new centroids
                from scipy.spatial.distance import cdist
                distances = cdist(object_centroids, input_centroids)
                
                # Find minimum distance for each existing object
                rows = distances.min(axis=1).argsort()
                cols = distances.argmin(axis=1)[rows]
                
                used_rows = set()
                used_cols = set()
                
                for row, col in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue
                        
                    if distances[row, col] > self.max_distance:
                        continue
                        
                    object_id = object_ids[row]
                    bbox, class_name, conf = detections[col]
                    
                    # Update object position and calculate velocity
                    obj = self.objects[object_id]
                    current_time = time.time()
                    new_pos = (*input_centroids[col], current_time)
                    
                    # Calculate velocity if we have previous position
                    if len(obj.positions) > 0:
                        prev_pos = obj.positions[-1]
                        dt = current_time - prev_pos[2]
                        if dt > 0:
                            vx = (new_pos[0] - prev_pos[0]) / dt
                            vy = (new_pos[1] - prev_pos[1]) / dt
                            obj.velocities.append((vx, vy))
                    
                    obj.positions.append(new_pos)
                    obj.last_bbox = bbox
                    obj.confidence = conf
                    self.disappeared[object_id] = 0
                    
                    used_rows.add(row)
                    used_cols.add(col)
                
                # Register new objects for unused detections
                unused_cols = set(range(len(detections))) - used_cols
                for col in unused_cols:
                    bbox, class_name, conf = detections[col]
                    self.register(input_centroids[col], bbox, class_name, conf)
                    
                # Mark unmatched objects as disappeared
                unused_rows = set(range(len(object_ids))) - used_rows
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
        return self.objects
    
    def detect_falls(self):
        """Analyze tracked objects for falling motion."""
        falling_objects = []
        
        for obj_id, obj in self.objects.items():
            if len(obj.velocities) < 2:
                continue
                
            # Get recent vertical velocities (in pixels/second)
            recent_velocities = list(obj.velocities)[-self.min_fall_frames:]
            if len(recent_velocities) < self.min_fall_frames:
                continue
                
            # Calculate average vertical velocity (positive = downward)
            avg_vy = np.mean([v[1] for v in recent_velocities])
            
            # Check if object is falling (consistent downward motion)
            if avg_vy > self.fall_velocity_threshold:
                if not obj.is_falling:
                    obj.is_falling = True
                    obj.fall_start_time = time.time()
                falling_objects.append(obj)
            else:
                obj.is_falling = False
                obj.fall_start_time = None
                
        return falling_objects

class DangerZone:
    """Manages the danger zone (rails area) definition and checking."""
    
    def __init__(self):
        self.points = []
        self.polygon = None
        self.is_defined = False
        
    def define_zone(self, frame):
        """Interactive danger zone definition."""
        print("\n=== DANGER ZONE DEFINITION ===")
        print("Click to define the rails area polygon.")
        print("Press 'f' when finished, 'c' to clear.")
        
        self.points = []
        clone = frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
                
        cv2.namedWindow("Define Danger Zone")
        cv2.setMouseCallback("Define Danger Zone", mouse_callback)
        
        while True:
            temp = clone.copy()
            
            # Draw points and polygon
            for i, point in enumerate(self.points):
                cv2.circle(temp, point, 5, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(temp, self.points[i-1], point, (0, 255, 0), 2)
            
            if len(self.points) > 2:
                cv2.polylines(temp, [np.array(self.points)], True, (0, 255, 0), 2)
                
            cv2.imshow("Define Danger Zone", temp)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('f') and len(self.points) > 2:
                self.polygon = np.array(self.points)
                self.is_defined = True
                break
            elif key == ord('c'):
                self.points = []
            elif key == 27:  # ESC
                break
                
        cv2.destroyWindow("Define Danger Zone")
        
        if self.is_defined:
            self.save_zone()
            print(f"Danger zone defined with {len(self.points)} points.")
        
    def is_in_zone(self, point):    
        """Check if a point is inside the danger zone."""
        if not self.is_defined or self.polygon is None:
            return False
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0
    
    def draw(self, frame, color=(0, 0, 255), alpha=0.3):
        """Draw the danger zone on the frame."""
        if not self.is_defined or self.polygon is None:
            return frame
            
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.polygon], color)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        cv2.polylines(frame, [self.polygon], True, color, 2)
        return frame
    
    def save_zone(self, filename="danger_zone.json"):
        """Save danger zone to file."""
        if self.is_defined:
            data = {"points": self.points}
            with open(filename, 'w') as f:
                json.dump(data, f)
    
    def load_zone(self, filename="danger_zone.json"):
        """Load danger zone from file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.points = data["points"]
                self.polygon = np.array(self.points)
                self.is_defined = True
                return True
        except FileNotFoundError:
            return False

def as_numpy(x):
    """Convert tensor/array to numpy array robustly."""
    try:
        return x.cpu().numpy()
    except Exception:
        return np.array(x)

def draw_tracked_object(frame, obj, color=(0, 200, 0), is_falling=False):
    """Draw bounding box and tracking info for an object."""
    x1, y1, x2, y2 = map(int, obj.last_bbox)
    
    # Change color if falling
    if is_falling:
        color = (0, 0, 255)  # Red for falling objects
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw object trail
    positions = list(obj.positions)
    for i in range(1, len(positions)):
        pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
        pt2 = (int(positions[i][0]), int(positions[i][1]))
        cv2.line(frame, pt1, pt2, color, 1)
    
    # Draw label
    label = f"ID:{obj.id} {obj.class_name} {obj.confidence:.2f}"
    if obj.is_falling:
        label += " FALLING!"
    
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def log_incident(obj, danger_zone):
    """Log fall incident to console and potentially to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get object position
    if len(obj.positions) > 0:
        last_pos = obj.positions[-1]
        x, y = last_pos[0], last_pos[1]
    else:
        x, y = 0, 0
    
    # Check if in danger zone
    in_zone = danger_zone.is_in_zone((x, y))
    
    if in_zone:
        print(f"\n🚨 ALERT: OBJECT FALL DETECTED IN DANGER ZONE!")
        print(f"   Time: {timestamp}")
        print(f"   Object: {obj.class_name} (ID: {obj.id})")
        print(f"   Position: ({x:.1f}, {y:.1f})")
        print(f"   Status: CRITICAL - Object in rails area!")
        print(f"   Action: Immediate response required!")
        
        # Log to file
        with open("fall_incidents.log", "a") as f:
            f.write(f"{timestamp},CRITICAL,{obj.class_name},{obj.id},{x:.1f},{y:.1f},IN_DANGER_ZONE\n")
    else:
        print(f"\n⚠️  WARNING: Object falling detected")
        print(f"   Time: {timestamp}")
        print(f"   Object: {obj.class_name} (ID: {obj.id})")
        print(f"   Position: ({x:.1f}, {y:.1f})")
        print(f"   Status: Monitoring - Not in danger zone")

def capture_incident_screenshot(frame, obj, danger_zone, in_danger_zone=False):
    """Capture and save a screenshot when a fall is detected."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename with incident details
    zone_status = "DANGER" if in_danger_zone else "WARNING"
    filename = f"incident_{timestamp}_{obj.class_name}_ID{obj.id}_{zone_status}.jpg"
    
    # File location
    filename = f"incident_images/{filename}"

    # Draw incident marker on the frame copy
    frame_copy = frame.copy()
    if len(obj.positions) > 0:
        x, y = int(obj.positions[-1][0]), int(obj.positions[-1][1])
        # Draw a circle at the falling object's position
        cv2.circle(frame_copy, (x, y), 20, (0, 0, 255), 3)
        cv2.putText(frame_copy, "FALL DETECTED", (x-50, y-25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Save the screenshot
    cv2.imwrite(filename, frame_copy)
    print(f"   📸 Screenshot saved: {filename}")
    
    return filename

def send_alert_to_backend(obj, image_path):
    """Send alert to backend system (stub function)."""
    # This function can be implemented to send HTTP requests or messages
    # to a backend monitoring system or alerting service.

    url = "http://ec2-54-84-92-63.compute-1.amazonaws.com/falldetection" 
    data = {
        "station": "Sevilla",
        "detected_object": obj.class_name,
        "incident_datetime": datetime.now().isoformat()
    }

    files = {
        'image': ('image.jpg', open(image_path, 'rb'), 'image/jpeg')
    }

    response = requests.post(url, data=data, files=files)
    print(response.status_code, response.text)
    

def main():
    parser = argparse.ArgumentParser(description="YOLO11 fall detection for rail safety monitoring.")
    parser.add_argument("--model", "-m", default="yolo11n.pt", help="Path or name of YOLO11 model")
    parser.add_argument("--device", "-d", default="cpu", help="Device for inference (cpu or cuda)")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--show-zone", action="store_true", help="Always show danger zone")
    args = parser.parse_args()

    # Check for scipy
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        print("ERROR: scipy is required. Install with: pip install scipy")
        sys.exit(1)

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(args.model)
    try:
        model.conf = args.conf
    except Exception:
        pass

    # Initialize components
    tracker = ObjectTracker(max_disappeared=15, max_distance=75)
    danger_zone = DangerZone()
    
    # Open camera
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2 if sys.platform.startswith("linux") else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    time.sleep(0.5)

    if not cap.isOpened():
        print("ERROR: Unable to open camera.", file=sys.stderr)
        sys.exit(1)

    window_name = "Fall Detection Monitor - ESC:exit, R:redefine zone, Z:toggle zone"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Try to load existing danger zone
    if danger_zone.load_zone():
        print("Loaded existing danger zone definition.")
    else:
        print("No danger zone found. Press 'r' to define the rails area.")
    
    # Get first frame for zone definition if needed
    ret, first_frame = cap.read()
    if ret and not danger_zone.is_defined:
        print("\n📍 Please define the danger zone (rails area) first!")
        danger_zone.define_zone(first_frame)
    
    show_zone = args.show_zone or True
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Empty frame.", file=sys.stderr)
                break
            
            frame_count += 1
            
            # Run YOLO inference
            results = model(frame, device=args.device, conf=args.conf, verbose=False)
            
            # Process detections
            detections = []
            if len(results) > 0:
                r = results[0]
                try:
                    xyxy = as_numpy(r.boxes.xyxy)
                    cls_ids = as_numpy(r.boxes.cls).astype(int)
                    confs = as_numpy(r.boxes.conf)
                    
                    for bbox, cid, conf in zip(xyxy, cls_ids, confs):
                        if int(cid) in TARGET_CLASS_IDS:
                            class_name = TARGET_CLASS_IDS[int(cid)]
                            detections.append((bbox, class_name, float(conf)))
                except Exception:
                    pass
            
            # Update tracker
            tracked_objects = tracker.update(detections)
            
            # Detect falls
            falling_objects = tracker.detect_falls()

            # Image location
            image_path = None
            
            # Check for new falls in danger zone
            for obj in falling_objects:
                if len(obj.positions) > 0:
                    current_pos = (obj.positions[-1][0], obj.positions[-1][1])
                    if danger_zone.is_in_zone(current_pos):
                        if not obj.entered_danger_zone:
                            obj.entered_danger_zone = True
                            log_incident(obj, danger_zone)
                            image_path = capture_incident_screenshot(frame, obj, danger_zone, in_danger_zone=True)
                            send_alert_to_backend(obj, image_path)
                    else:
                        if obj.fall_start_time and (time.time() - obj.fall_start_time) > 0.5:
                            # Log falls outside danger zone after 0.5 seconds
                            if not obj.entered_danger_zone:
                                log_incident(obj, danger_zone)
                                image_path = capture_incident_screenshot(frame, obj, danger_zone, in_danger_zone=False)
                                send_alert_to_backend(obj, image_path)
                                obj.entered_danger_zone = True
            
            # Draw danger zone
            if show_zone:
                frame = danger_zone.draw(frame)
            
            # Draw tracked objects
            for obj_id, obj in tracked_objects.items():
                is_falling = obj in falling_objects
                draw_tracked_object(frame, obj, is_falling=is_falling)
            
            # Display stats
            stats_text = f"Tracking: {len(tracked_objects)} | Falling: {len(falling_objects)}"
            cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if not danger_zone.is_defined:
                cv2.putText(frame, "Press 'R' to define danger zone", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow(window_name, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('R'):
                danger_zone.define_zone(frame)
            elif key == ord('z') or key == ord('Z'):
                show_zone = not show_zone

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nFall detection system stopped.")

if __name__ == "__main__":
    main()
