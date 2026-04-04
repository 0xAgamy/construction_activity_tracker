
"""
Enhanced CV pipeline:
  - YOLO detection
  - DeepSort multi-object tracking
  - Region-based optical flow (handles articulated motion)
  - Activity classification
  - Kafka streaming
"""

import cv2
import json
import time
import uuid
import os
import numpy as np
from collections import defaultdict, deque
import csv

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import helpers
from activity_classifier import ActivityClassifier

from denseflow import OpticalFlowEngine
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC     =  "equipment-events"
SESSION_ID      = str(uuid.uuid4())[:8]
COLLECT_TRAINING_DATA=True

STATE_DEBOUNCE_FRAMES = 10     # Majority-vote window for state smoothing


# Target classes (from your trained model)
TARGET_CLASSES = {
    0: "Dump truck",     1: "Excavator",         2: "Motor grader",
    3: "Roller",         4: "Crane manipulator", 5: "Gazelle",
    6: "Forklift Standart", 7: "Bucket loader Big", 8: "Mixer",
    9: "Tanker",         10: "Bulldozer",         11: "Cleaning equipment",
    12: "Truck",         13: "Trailer",            14: "Forklift Giraffe",
    15: "Bucket loader Standart", 16: "Autocran"
}

class UtilizationTracker:
    """
    Accumulates ACTIVE / INACTIVE time per tracked object.
    Uses real wall-clock seconds between consecutive updates.
    """

    def __init__(self):
        # { track_id: {"active": float, "inactive": float, "last_ts": float} }
        self._data: dict[int, dict] = defaultdict(lambda: {
            "active": 0.0, "inactive": 0.0,
            "last_ts": None, "class_name": ""
        })

    def update(self, track_id: int, state: str,
               class_name: str, timestamp: float) -> dict:
        rec = self._data[track_id]
        rec["class_name"] = class_name

        if rec["last_ts"] is not None:
            delta = timestamp - rec["last_ts"]
            # Guard against clock jumps
            delta = min(delta, 5.0)
            if state == "ACTIVE":
                rec["active"] += delta
            else:
                rec["inactive"] += delta

        rec["last_ts"] = timestamp

        total = rec["active"] + rec["inactive"]
        utilization = (rec["active"] / total * 100.0) if total > 0 else 0.0

        return {
            "total_active_sec":   round(rec["active"],    2),
            "total_inactive_sec": round(rec["inactive"],  2),
            "total_tracked_sec":  round(total,            2),
            "utilization_pct":    round(utilization,      2)
        }

    def get_summary(self) -> list[dict]:
        rows = []
        for tid, rec in self._data.items():
            total = rec["active"] + rec["inactive"]
            rows.append({
                "track_id":   tid,
                "class_name": rec["class_name"],
                "active_sec": round(rec["active"],   2),
                "idle_sec":   round(rec["inactive"], 2),
                "total_sec":  round(total,           2),
                "util_pct":   round(rec["active"] / total * 100 if total else 0, 2)
            })
        return rows


def create_kafka_producer(retries: int = 10, delay: int = 5) -> KafkaProducer:
    for attempt in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BOOTSTRAP],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                # Reliability settings
                acks="all",
                retries=3,
                linger_ms=10,        # Small batching window
                batch_size=16384
            )
            print(f"✅ Kafka producer connected to {KAFKA_BOOTSTRAP}")
            return producer
        except NoBrokersAvailable:
            print(f"⏳ Kafka not ready (attempt {attempt+1}/{retries}). "
                  f"Retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError("❌ Could not connect to Kafka after retries.")


class StateDebouncer:
    """
    Prevents rapid flickering between ACTIVE/INACTIVE
    """

    def __init__(self, window: int = STATE_DEBOUNCE_FRAMES):
        self._window = window
        self._histories: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def update(self, track_id: int, raw_state: str) -> str:
        buf = self._histories[track_id]
        buf.append(raw_state)
        active_count = buf.count("ACTIVE")
        return "ACTIVE" if active_count > len(buf) // 2 else "INACTIVE"


frames=[]

def process_video(video_path: str):
    print(f"Loading video: {video_path}")
    print(f"Kafka: {KAFKA_BOOTSTRAP} → topic: {KAFKA_TOPIC}")
    print(f"Session ID: {SESSION_ID}")

    # ── Initialise components ──────────────────────────────────────
    model    = YOLO("models/best.pt")
    model.to("cpu")

    tracker  = DeepSort(max_age=30, embedder_gpu=False)
    flow_eng = OpticalFlowEngine()
    debounce = StateDebouncer()
    act_cls  = ActivityClassifier()
    util_trk = UtilizationTracker()
    producer = create_kafka_producer()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = 0

    print("Processing started…")

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            ts = time.time()

            # ── Optical flow ───────────────────────────────────────
            gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray      = cv2.GaussianBlur(gray, (7, 7), 0)
            flow_mag, adapt_thresh = flow_eng.update(gray)
            flow_vec = flow_eng.get_flow_vectors(gray) 
            #  YOLO detection 
            results = model(frame, verbose=False)[0]
            dets    = helpers.parse_detections(results)

            #  DeepSort tracking 
            tracks = tracker.update_tracks(dets, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                tid        = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                class_name = str(
                    model.names.get(track.det_class, "Unknown")
                    if track.det_class is not None else "Unknown"
                )
                confidence = float(track.det_conf or 0.0)

                #  Motion analysis 
                motion = flow_eng.analyse_bbox(
                    flow_mag, (x1, y1, x2, y2), adapt_thresh
                ) if flow_mag is not None else OpticalFlowEngine._empty_metrics()

                #  State with debouncing 
                state = debounce.update(tid, motion["raw_state"])

                #  Activity classification 
                activity = act_cls.update(
                    tid, class_name,
                    (x1, y1, x2, y2),
                    flow_mag,
                    flow_vec,
                    state
                )

                #  Utilization accumulation 
                util = util_trk.update(tid, state, class_name, ts)

                # Build Kafka payload
                payload = {
                    # Identity
                    "track_id":   int(tid),
                    "session_id": SESSION_ID,
                    "timestamp":  round(ts, 3),
                    "frame":      int(frame_count),

                    # Classification
                    "class_name": class_name,
                    "state":      state,
                    "activity":   activity,
                    "confidence": round(confidence, 3),

                    # Bounding box
                    "bbox": {
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2)
                    },

                    # Utilization
                    "utilization": util,

                    # Debug
                    "debug": {
                        "motion_score":    round(motion["avg_motion"],  3),
                        "max_region_flow": round(motion["max_region"],  3),
                        "active_pixels":   int(motion["active_pixels"]),
                        "adaptive_thresh": round(adapt_thresh, 3),
                        "calibrated":      flow_eng.calibrated
                    }
                }

                #  Send to Kafka 
                try:
                    producer.send(KAFKA_TOPIC, value=payload)
                except Exception as e:
                    print(f"  Kafka send error: {e}")

                # After send the pyalod to kafka i will save the output/payload for later use
                if COLLECT_TRAINING_DATA:
                    with open("outputs/all_predictions.csv", 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=payload.keys())
                        if f.tell() == 0:
                            writer.writeheader()
                        writer.writerow(payload)
                # ── Annotate frame ─────────────────────────────────
                helpers.draw_annotations(frame, payload)


            # save data for training
            # ── Display ────────────────────────────────────────────
            frames.append(frame)
            cv2.imshow("Construction Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

            if frame_count % 100 == 0:
                print(f"  Frame {frame_count} | "
                      f"Tracks: {len([t for t in tracks if t.is_confirmed()])}")

    finally:
        helpers.video_saver(frames)
        cap.release()
        cv2.destroyAllWindows()
        producer.flush()
        producer.close()

        # Print final summary
        print("\nFinal Utilization Summary:")
        print(f"{'Track':>6}  {'Class':<24} {'Active':>8}  "
              f"{'Idle':>8}  {'Total':>8}  {'Util%':>7}")
        print("-" * 70)
        for row in util_trk.get_summary():
            print(f"{row['track_id']:>6}  {row['class_name']:<24} "
                  f"{row['active_sec']:>8.1f}  {row['idle_sec']:>8.1f}  "
                  f"{row['total_sec']:>8.1f}  {row['util_pct']:>6.1f}%")

        print(f"\nDone — {frame_count} frames processed.")


