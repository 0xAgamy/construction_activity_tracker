import numpy as np
from collections import defaultdict, deque
import cv2
from typing import Optional
FLOW_THRESHOLD= 2.0    
MIN_ACTIVE_PIXELS= 10     
CALIBRATION_FRAMES= 30 
NOISE_MULTIPLIER= 2.0  

class OpticalFlowEngine:
    """
    Dense optical flow (Farneback) with adaptive noise calibration.
    Handles articulated motion: analyses sub-regions of the bounding box
    to detect motion even when only part of the machine moves.
    """

    def __init__(self):
        self.prev_gray:   np.ndarray | None = None
        self._noise_buf:  deque = deque(maxlen=CALIBRATION_FRAMES)
        self.calibrated:  bool  = False
        self.noise_floor: float = FLOW_THRESHOLD

    def update(self, frame_gray: np.ndarray
               ) -> tuple[np.ndarray | None, float]:
        """
        Compute flow, update calibration.
        Returns (flow_magnitude_map, adaptive_threshold).
        """
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return None, FLOW_THRESHOLD

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, frame_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag = cv2.magnitude(flow[..., 0], flow[..., 1])

        # measure background noise level
        if not self.calibrated:
            self._noise_buf.append(float(np.mean(mag)))
            if len(self._noise_buf) >= CALIBRATION_FRAMES:
                self.noise_floor = float(np.mean(self._noise_buf)) * NOISE_MULTIPLIER
                self.calibrated  = True
                print(f"Flow calibrated — noise floor: {self.noise_floor:.3f}")

        self.prev_gray = frame_gray
        adaptive_thresh = self.noise_floor if self.calibrated else FLOW_THRESHOLD
        return mag, adaptive_thresh

    def analyse_bbox(
        self,
        flow_mag: np.ndarray,
        bbox: tuple[int, int, int, int],
        adaptive_thresh: float
    ) -> dict:
        """
        Analyse optical flow within a bounding box.
        Uses multi-region analysis to detect articulated motion
       

        Returns motion metrics dict.
        """
        x1, y1, x2, y2 = bbox
        fh, fw = flow_mag.shape

        # Clamp to frame
        x1c, x2c = max(0, x1), min(fw, x2)
        y1c, y2c = max(0, y1), min(fh, y2)

        if x2c <= x1c or y2c <= y1c:
            return self._empty_metrics()

        roi = flow_mag[y1c:y2c, x1c:x2c]
        if roi.size == 0:
            return self._empty_metrics()

        rh, rw = roi.shape

        #  Sub-region decomposition (3×3 grid)
        # Even if 8/9 regions are still, one active region = ACTIVE machine
        thirds_h = [0, rh//3, 2*rh//3, rh]
        thirds_w = [0, rw//3, 2*rw//3, rw]

        region_scores = []
        for i in range(3):
            for j in range(3):
                cell = roi[thirds_h[i]:thirds_h[i+1],
                           thirds_w[j]:thirds_w[j+1]]
                if cell.size > 0:
                    region_scores.append(float(np.mean(cell)))

        active_mask   = roi > adaptive_thresh
        active_pixels = int(np.sum(active_mask))
        avg_motion    = float(np.mean(roi))
        max_region    = max(region_scores) if region_scores else 0.0

        
        any_region_active = max_region > adaptive_thresh
        enough_pixels     = active_pixels >= MIN_ACTIVE_PIXELS

        is_active = any_region_active and enough_pixels

        return {
            "raw_state":     "ACTIVE" if is_active else "INACTIVE",
            "avg_motion":    avg_motion,
            "max_region":    max_region,
            "active_pixels": active_pixels,
            "region_scores": region_scores
        }

    def get_flow_vectors(self, frame_gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns dense optical flow vectors (dx, dy) for direction analysis.
        Shape: (height, width, 2)
        """
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, frame_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        self.prev_gray = frame_gray
        return flow  # Shape: (H, W, 2)
    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "raw_state":     "INACTIVE",
            "avg_motion":    0.0,
            "max_region":    0.0,
            "active_pixels": 0,
            "region_scores": []
        }

