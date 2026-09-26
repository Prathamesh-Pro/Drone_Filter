import cv2
import mediapipe as mp

from Hand_Pos import INDEX_TIP, THUMB_TIP, WRIST, MIDDLE_TIP
from Shape import render_portal, portal_width, ClosingGestureDetector
from Drone_filter import FILTERS, FILTER_NAMES
from Motion_Detector import MotionDetector
from Weather_History import WeatherHistoryMonitor
from Distance_Estimator import estimate_distance_cm, estimate_real_area_cm2
from Kalman_Tracker import PointKalmanFilter


def clamp_point(point, w, h):
    x = min(max(point[0], 0), w - 1)
    y = min(max(point[1], 0), h - 1)
    return (x, y)


PROCESS_SCALE = 0.5  # hand tracking runs on a downscaled frame for speed; display stays full-res


def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the camera. Check the camera index or permissions."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # drop stale buffered frames to cut input lag

    filter_index = 0
    closing_detector = ClosingGestureDetector()
    motion_detector = MotionDetector()
    motion_enabled = False
    weather_monitor = WeatherHistoryMonitor()

    point_trackers = {
        "p1": PointKalmanFilter(),
        "p2": PointKalmanFilter(),
        "p3": PointKalmanFilter(),
        "p4": PointKalmanFilter(),
    }

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        small_frame = cv2.resize(frame, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left_hand = None
        right_hand = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                raw_label = handedness.classification[0].label
                label = "Right" if raw_label == "Left" else "Left"

                if label == "Left":
                    left_hand = hand_landmarks
                else:
                    right_hand = hand_landmarks

        if left_hand is not None and right_hand is not None:
            lm_left = left_hand.landmark
            lm_right = right_hand.landmark

            p1 = clamp_point((lm_left[INDEX_TIP].x * w, lm_left[INDEX_TIP].y * h), w, h)
            p2 = clamp_point((lm_left[THUMB_TIP].x * w, lm_left[THUMB_TIP].y * h), w, h)
            p3 = clamp_point((lm_right[INDEX_TIP].x * w, lm_right[INDEX_TIP].y * h), w, h)
            p4 = clamp_point((lm_right[THUMB_TIP].x * w, lm_right[THUMB_TIP].y * h), w, h)

            p1 = point_trackers["p1"].update(p1)
            p2 = point_trackers["p2"].update(p2)
            p3 = point_trackers["p3"].update(p3)
            p4 = point_trackers["p4"].update(p4)

            wrist_px = clamp_point((lm_right[WRIST].x * w, lm_right[WRIST].y * h), w, h)
            middle_tip_px = clamp_point(
                (lm_right[MIDDLE_TIP].x * w, lm_right[MIDDLE_TIP].y * h), w, h
            )
            distance_cm = estimate_distance_cm(wrist_px, middle_tip_px, w)

            width = portal_width(p1, p2, p3, p4)

            if closing_detector.update(width, w):
                filter_index = (filter_index + 1) % len(FILTERS)
                filter_name = FILTER_NAMES[FILTERS[filter_index]]
                print(f"Filter changed: {filter_name}")

            frame = render_portal(frame, p1, p2, p3, p4, FILTERS[filter_index])

            top_left = (int(min(p1[0], p2[0])), int(min(p1[1], p2[1])) - 10)
            cv2.putText(
                frame,
                FILTER_NAMES[FILTERS[filter_index]],
                top_left,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            bottom_left = (int(min(p1[0], p2[0])), int(max(p3[1], p4[1])) + 25)
            if distance_cm is not None:
                area_cm2 = estimate_real_area_cm2([p1, p3, p4, p2], distance_cm, w)
                dist_text = f"Distance: {distance_cm:.1f} cm"
                area_text = f"Area: {area_cm2:.1f} cm^2 (approx)"
            else:
                dist_text = "Distance: N/A"
                area_text = "Area: N/A"
            cv2.putText(
                frame,
                dist_text,
                bottom_left,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                area_text,
                (bottom_left[0], bottom_left[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
        else:
            for tracker in point_trackers.values():
                tracker.reset()

        if motion_enabled:
            boxes = motion_detector.detect(frame)
            frame = motion_detector.draw(frame, boxes)

        cv2.putText(
            frame,
            weather_monitor.get_status_text(),
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        cv2.imshow("Drone Filters", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            motion_enabled = not motion_enabled

    weather_monitor.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()