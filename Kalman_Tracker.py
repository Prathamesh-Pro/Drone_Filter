import cv2
import numpy as np


class PointKalmanFilter:
    """Smooths a single 2D point (e.g. a fingertip landmark) using a
    constant-velocity Kalman filter. Reduces per-frame detection jitter
    without adding lag, since it predicts + corrects every frame instead
    of averaging over a delay window."""

    def __init__(self, process_noise=1e-2, measurement_noise=5e-1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.initialized = False

    def update(self, point):
        if not self.initialized:
            self.kf.statePre = np.array(
                [[point[0]], [point[1]], [0], [0]], dtype=np.float32
            )
            self.kf.statePost = self.kf.statePre.copy()
            self.initialized = True
            return point

        self.kf.predict()
        measurement = np.array(
            [[np.float32(point[0])], [np.float32(point[1])]]
        )
        corrected = self.kf.correct(measurement)
        return (float(corrected[0]), float(corrected[1]))

    def reset(self):
        self.initialized = False
