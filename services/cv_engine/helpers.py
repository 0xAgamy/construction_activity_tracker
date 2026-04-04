
import cv2
import numpy as np
import os,csv
def parse_detections(results)->list:
    """convert yolo results to detections"""

    boxes= results.boxes.xyxy.cpu().numpy()
    confs= results.boxes.conf.cpu().numpy()
    cls_ids= results.boxes.cls.int().cpu().numpy()
    detections= []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2= box
        detections.append(([x1, y1, x2 - x1, y2 - y1], confs[i], cls_ids[i]))
    return detections

def draw_annotations(frame: np.ndarray, payload: dict) -> None:
    """Draw bounding box and status overlay on frame."""
    bbox       = payload["bbox"]
    state      = payload["state"]
    activity   = payload["activity"]
    class_name = payload["class_name"]
    tid        = payload["track_id"]
    util       = payload["utilization"]

    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

    # green = ACTIVE, red = INACTIVE
    color = (0, 255, 0) if state == "ACTIVE" else (0, 0, 255)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    label = f"{class_name} #{tid} | {state} | {activity}"
    util_label = (f"A:{util['total_active_sec']:.0f}s "
                  f"I:{util['total_inactive_sec']:.0f}s "
                  f"U:{util['utilization_pct']:.1f}%")

    for i, text in enumerate([label, util_label]):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y1 - 10 - (th + 4) * i
        cv2.rectangle(frame, (x1, ty - th - 2), (x1 + tw, ty + 2),
                      (0, 0, 0), -1)
        cv2.putText(frame, text, (x1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
def video_saver(frames:np.ndarray):
    """i Create this function to better output video analysis
    """
    h, w ,l= frames[0].shape

    forcc=cv2.VideoWriter_fourcc(*'mp4v') 
    out= cv2.VideoWriter("outputs/output.mp4",forcc,30,(w,h))
    for frame in frames:
        out.write(frame)
    
    out.release()




