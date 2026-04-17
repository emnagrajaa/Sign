import cv2
import mediapipe as mp
import numpy as np

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])

            print(f"Extracted {len(keypoints)} landmarks")

            normalized = normalize_keypoints(keypoints)
            print("Normalized shape:", normalized.shape)

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

    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
