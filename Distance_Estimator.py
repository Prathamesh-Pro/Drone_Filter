import numpy as np

ASSUMED_HAND_LENGTH_CM = 18.0
ASSUMED_HFOV_DEG = 60.0


def focal_length_px(frame_width):
    hfov_rad = np.radians(ASSUMED_HFOV_DEG)
    return frame_width / (2 * np.tan(hfov_rad / 2))


def estimate_distance_cm(wrist_px, middle_tip_px, frame_width):
    hand_length_px = np.hypot(
        wrist_px[0] - middle_tip_px[0], wrist_px[1] - middle_tip_px[1]
    )
    if hand_length_px < 1e-6:
        return None
    f_px = focal_length_px(frame_width)
    return (ASSUMED_HAND_LENGTH_CM * f_px) / hand_length_px


def polygon_area_px(points):
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def estimate_real_area_cm2(points, distance_cm, frame_width):
    area_px = polygon_area_px(points)
    f_px = focal_length_px(frame_width)
    cm_per_px = distance_cm / f_px
    return area_px * (cm_per_px ** 2)
