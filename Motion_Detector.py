import cv2
import numpy as np


class MotionDetector:
    """Background-subtraction based motion/intrusion detector."""

    def __init__(self, history=500, var_threshold=40, min_area=800):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )
        self.min_area = min_area
        self.kernel = np.ones((3, 3), np.uint8)

    def detect(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = [
            cv2.boundingRect(c)
            for c in contours
            if cv2.contourArea(c) >= self.min_area
        ]
        return boxes

    def draw(self, frame, boxes):
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if boxes:
            cv2.putText(
                frame,
                f"MOTION DETECTED ({len(boxes)})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        return frame
