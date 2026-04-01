from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
from ultralytics import YOLO
import json
import time
import math
from kafka import KafkaProducer
import pandas as pd
import numpy as np

model= YOLO("models/best.pt")
device = 'cpu'
model.to(device)

KAFKA_TOPIC = "equipment-events"
FLOW_THRESHOLD = 2.0                # Raw flow value (not normalized)
MIN_ACTIVE_PIXELS = 10              # At least N pixels must be moving
STATE_DEBOUNCE_FRAMES = 5           # Require N frames to change state
CALIBRATION_FRAMES = 10            
                     
TARGET_CLASSES   = {
0: "Dump truck",
1: "Excavator",
2: "Motor grader",
3: "Roller",
4: "Crane manipulator",
5: "Gazelle",
6: "Forklift Standart",
7: "Bucket loader Big",
8: "Mixer",
9: "Tanker",
10: "Bulldozer",
11: "Cleaning equipment",
12: "Truck",
13: "Trailer",
14: "Forklift Giraffe",
15: "Bucket loader Standart",
16: "Autocran"

}

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

tracker = DeepSort(max_age=30, embedder_gpu=False)


# For Optical Flow
prev_gray = None
flow_history = []
calibration_complete = False


state_history = {}

def get_dominant_state(history):
    if not history:
        return "INACTIVE"
    return "ACTIVE" if history.count("ACTIVE") > len(history) // 2 else "INACTIVE"

def process_video(VIDEO_PATH):
    global prev_gray, flow_history, calibration_complete
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("❌ Cannot open video file")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    print("🚀 Starting Video Processing...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, (7, 7), 0)

        # 2. Detection
        results = model(frame, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.int().cpu().numpy()

        
        detections = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], confs[i], class_ids[i]))

        
        tracks = tracker.update_tracks(detections, frame=frame)

        # Calculate Optical Flow
        flow_magnitude_map = None
        
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                                pyr_scale=0.5, levels=3, 
                                                winsize=15, iterations=3, 
                                                poly_n=5, poly_sigma=1.2, 
                                                flags=0)
            flow_magnitude_map = cv2.magnitude(flow[..., 0], flow[..., 1])
            
            # Calibration
            if not calibration_complete:
                baseline_noise = np.mean(flow_magnitude_map)
                flow_history.append(baseline_noise)
                
                if len(flow_history) >= CALIBRATION_FRAMES:
                    calibration_complete = True
                    avg_baseline = np.mean(flow_history)
                    print(f"✅ Calibration Complete. Baseline Noise: {avg_baseline:.2f}")

        
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            class_name = model.names[track.det_class]
            
            # 🔹 Determine State
            raw_state = "INACTIVE"
            avg_motion = 0.0
            active_pixels = 0
            
            if flow_magnitude_map is not None:
                y1_clamped = max(0, y1)
                y2_clamped = min(flow_magnitude_map.shape[0], y2)
                x1_clamped = max(0, x1)
                x2_clamped = min(flow_magnitude_map.shape[1], x2)
                
                if y2_clamped > y1_clamped and x2_clamped > x1_clamped:
                    roi_flow = flow_magnitude_map[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
                    
                    if roi_flow.size > 0:
                        adaptive_threshold = FLOW_THRESHOLD
                        if calibration_complete:
                            adaptive_threshold = np.mean(flow_history) * 2
                        
                        active_mask = roi_flow > adaptive_threshold
                        active_pixels = int(np.sum(active_mask))  # 🔹 Cast to int
                        avg_motion = float(np.mean(roi_flow))     # 🔹 Cast to float
                        
                        if avg_motion > adaptive_threshold and active_pixels > MIN_ACTIVE_PIXELS:
                            raw_state = "ACTIVE"
            
            # State Debouncing
            if track_id not in state_history:
                state_history[track_id] = []
            
            state_history[track_id].append(raw_state)
            if len(state_history[track_id]) > STATE_DEBOUNCE_FRAMES:
                state_history[track_id].pop(0)
            
            state = get_dominant_state(state_history[track_id])

            # 7. Create Payload (🔹 ALL VALUES CAST TO NATIVE TYPES)
            payload = {
                "track_id": int(track_id),
                "timestamp": float(time.time()),
                "frame": int(frame_count),
                "class_name": str(class_name),
                "state": str(state),
                "activity": str("Working"),
                "confidence": float(track.det_conf) if track.det_conf else 0.0,
                "bbox": {
                    "x1": int(x1), 
                    "y1": int(y1), 
                    "x2": int(x2), 
                    "y2": int(y2)
                },
                "debug_motion_score": float(avg_motion),
                "debug_active_pixels": int(active_pixels)
            }

            # Send to Kafka
            try:
                producer.send(KAFKA_TOPIC, value=payload)
            except Exception as e:
                print(f"❌ Kafka Send Error: {e}")

            # Visualize
            color = (0, 255, 0) if state == "ACTIVE" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name} {track_id} [{state}] M:{avg_motion:.2f} P:{active_pixels}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update previous frame
        prev_gray = curr_gray

        # Show Video Window - i think i will just save and output video for better checking
        cv2.imshow("Construction Monitor (Calibrated Flow)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    producer.close()
    print("✅ Processing Complete.")