import cv2
import mediapipe as mp
import numpy as np
import time

CAM_WIDTH = 960
CAM_HEIGHT = 540
SHOW_LANDMARK_INDEX = False
STATUS_PRINT_INTERVAL_SEC = 1.0

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


def normalize_keypoints(keypoints):
    keypoints = np.array(keypoints, dtype=np.float32)

    # Use the wrist as the origin so hand position in the frame does not matter.
    wrist = keypoints[0]
    keypoints -= wrist

    # Scale relative to the largest axis distance so hand size is normalized too.
    max_dist = np.max(np.abs(keypoints))
    if max_dist > 0:
        keypoints /= max_dist

    return keypoints.flatten()


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    hands.close()
    raise RuntimeError(
        "Could not open webcam. Make sure camera access is enabled and no "
        "other app is using it."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

ema_fps = 0.0
last_status_print = time.perf_counter()

while True:
    loop_start = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    after_read = time.perf_counter()

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    after_mediapipe = time.perf_counter()
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            keypoints = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

            normalized = normalize_keypoints(keypoints)
            if SHOW_LANDMARK_INDEX:
                for idx, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.putText(
                        frame,
                        str(idx),
                        (cx - 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1
                    )

    loop_end = time.perf_counter()
    frame_time = max(loop_end - loop_start, 1e-6)
    instant_fps = 1.0 / frame_time
    ema_fps = instant_fps if ema_fps == 0.0 else (0.9 * ema_fps + 0.1 * instant_fps)

    read_ms = (after_read - loop_start) * 1000.0
    mediapipe_ms = (after_mediapipe - after_read) * 1000.0
    total_ms = frame_time * 1000.0

    cv2.rectangle(frame, (0, 0), (w, 64), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"FPS: {ema_fps:5.1f}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 120),
        2,
    )
    cv2.putText(
        frame,
        f"read {read_ms:4.1f}ms  mediapipe {mediapipe_ms:4.1f}ms  total {total_ms:4.1f}ms",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (190, 190, 190),
        1,
    )

    now = time.perf_counter()
    if now - last_status_print >= STATUS_PRINT_INTERVAL_SEC:
        print(
            f"fps={ema_fps:5.1f} read={read_ms:4.1f}ms "
            f"mediapipe={mediapipe_ms:4.1f}ms total={total_ms:4.1f}ms"
        )
        last_status_print = now

    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
