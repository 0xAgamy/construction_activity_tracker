# activity_classifier.py
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple
import cv2

@dataclass
class ActivityStatus:
    track_id: int
    class_name: str
    upper_motion: deque = field(default_factory=lambda: deque(maxlen=15))
    lower_motion: deque = field(default_factory=lambda: deque(maxlen=15))
    left_motion: deque = field(default_factory=lambda: deque(maxlen=15))
    right_motion: deque = field(default_factory=lambda: deque(maxlen=15))
    vertical_flow: deque = field(default_factory=lambda: deque(maxlen=15))
    horizontal_flow: deque = field(default_factory=lambda: deque(maxlen=15))
    activities: deque = field(default_factory=lambda: deque(maxlen=10))

# ── TUNABLE THRESHOLDS ──────────────────────────────────────────
DIG_THRESHOLD       = 1.5    # Lower part motion for digging
SWING_THRESHOLD     = 1.8    # Horizontal asymmetry for swinging
DUMP_THRESHOLD      = 2.0    # Upper part motion for dumping
ACTIVE_THRESHOLD    = 0.8    # Minimum flow for any activity
VERTICAL_THRESHOLD  = 1.2    # Vertical flow for dig/dump
HORIZONTAL_THRESHOLD= 1.5    # Horizontal flow for swinging
# ────────────────────────────────────────────────────────────────

class ActivityClassifier:
    def __init__(self):
        self.status: dict[int, ActivityStatus] = {}

    def create_or_get(self, track_id: int, class_name: str) -> ActivityStatus:
        if track_id not in self.status:
            self.status[track_id] = ActivityStatus(
                track_id=track_id,
                class_name=class_name
            )
        return self.status[track_id]

    def update(self,
               track_id: int,
               class_name: str,
               bbox: Tuple[int, int, int, int],
               flow_magnitude_map: Optional[np.ndarray],
               flow_vector_map: Optional[np.ndarray],  # NEW: Direction info
               state: str) -> str:
        
        activity_state = self.create_or_get(track_id, class_name)

        # Exit if no flow data or inactive
        if flow_magnitude_map is None or state == "INACTIVE":
            activity_state.activities.append("Waiting")
            return "Waiting"

        x1, y1, x2, y2 = bbox
        h, w = y2 - y1, x2 - x1
        
        if h <= 0 or w <= 0:
            return "Waiting"
        
        # Clamp to frame boundaries
        fh, fw = flow_magnitude_map.shape
        x1c, x2c = max(0, x1), min(fw, x2)
        y1c, y2c = max(0, y1), min(fh, y2)

        roi_mag = flow_magnitude_map[y1c:y2c, x1c:x2c]
        
        if roi_mag.size == 0:
            return "Waiting"

        roi_h, roi_w = roi_mag.shape

        # Split ROI into regions
        upper_cut = int(roi_h * 0.4)
        lower_cut = int(roi_h * 0.6)
        left_cut = int(roi_w * 0.5)

        upper_region = roi_mag[:upper_cut, :]
        lower_region = roi_mag[lower_cut:, :]
        left_region = roi_mag[:, :left_cut]
        right_region = roi_mag[:, left_cut:]

        # Calculate region means
        upper_mean = float(np.mean(upper_region)) if upper_region.size > 0 else 0.0
        lower_mean = float(np.mean(lower_region)) if lower_region.size > 0 else 0.0
        left_mean = float(np.mean(left_region)) if left_region.size > 0 else 0.0
        right_mean = float(np.mean(right_region)) if right_region.size > 0 else 0.0

        # NEW: Calculate flow direction (vertical vs horizontal)
        vertical_mean = 0.0
        horizontal_mean = 0.0
        
        if flow_vector_map is not None:
            roi_flow = flow_vector_map[y1c:y2c, x1c:x2c]
            if roi_flow.size > 0:
                # flow_vector_map shape: (H, W, 2) where [:, :, 0]=dx, [:, :, 1]=dy
                vertical_mean = float(np.mean(np.abs(roi_flow[:, :, 1])))  # dy
                horizontal_mean = float(np.mean(np.abs(roi_flow[:, :, 0])))  # dx

        # Update history
        activity_state.upper_motion.append(upper_mean)
        activity_state.lower_motion.append(lower_mean)
        activity_state.left_motion.append(left_mean)
        activity_state.right_motion.append(right_mean)
        activity_state.vertical_flow.append(vertical_mean)
        activity_state.horizontal_flow.append(horizontal_mean)

        # Classify activity
        activity = self._classify(activity_state, class_name)
        activity_state.activities.append(activity)

        # Smooth with recent history (last 5 predictions)
        recent = list(activity_state.activities)[-5:]
        return max(set(recent), key=recent.count)

    def _classify(self, s: ActivityStatus, class_name: str) -> str:
        if len(s.upper_motion) < 3:
            return "Working"

        avg_upper = np.mean(s.upper_motion)
        avg_lower = np.mean(s.lower_motion)
        avg_left = np.mean(s.left_motion)
        avg_right = np.mean(s.right_motion)
        avg_vertical = np.mean(s.vertical_flow)
        avg_horizontal = np.mean(s.horizontal_flow)
        
        total_flow = (avg_upper + avg_lower) / 2.0
        horiz_asymmetry = abs(avg_left - avg_right)

        # ── Excavator Rules ─────────────────────────────────────
        if class_name == "Excavator":
            # Dumping: Strong upward motion in upper region
            if avg_upper > DUMP_THRESHOLD and avg_upper > avg_lower * 1.4:
                return "Dumping"
            
            # Digging: Strong downward motion in lower region
            if avg_lower > DIG_THRESHOLD and avg_lower > avg_upper * 1.2:
                return "Digging"
            
            # Swinging: High horizontal asymmetry (left vs right)
            if horiz_asymmetry > SWING_THRESHOLD and total_flow > ACTIVE_THRESHOLD:
                return "Swinging/Loading"
            
            # Vertical-dominant motion = Digging or Dumping
            if avg_vertical > VERTICAL_THRESHOLD and avg_vertical > avg_horizontal:
                if avg_lower > avg_upper:
                    return "Digging"
                else:
                    return "Dumping"
            
            # Default active excavator
            if total_flow > ACTIVE_THRESHOLD:
                return "Digging"

        # ── Dump Truck Rules ────────────────────────────────────
        elif class_name == "Dump truck":
            if avg_upper > DUMP_THRESHOLD:
                return "Dumping"
            if total_flow > ACTIVE_THRESHOLD:
                return "Working"

        # ── Default ─────────────────────────────────────────────
        if total_flow > ACTIVE_THRESHOLD:
            return "Working"

        return "Waiting"