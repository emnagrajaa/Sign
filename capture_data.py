import csv
import os
import time

import cv2
import mediapipe as mp
import numpy as np

# Config
GESTURES = [
    "hello",
    "stop",
    "yes",
    "no",
    "point",
    "peace",
    "iloveyou",
    "call",
    "ok",
    "four",
]
SAMPLES_PER_GESTURE = 150
COUNTDOWN_SEC = 3
OUTPUT_CSV = "data/gestures.csv"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


def normalize_keypoints(landmarks):
    kp = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    kp -= kp[0]
    scale = np.max(np.abs(kp))
    if scale > 0:
        kp /= scale
    return kp.flatten().tolist()


def draw_ui(frame, gesture, collected, total, state, countdown=0):
    _, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)

    cv2.putText(
        frame,
        f"Gesture: {gesture.upper()}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    if state == "countdown":
        msg = f"Get ready... {countdown}"
        color = (0, 200, 255)
    elif state == "recording":
        msg = f"Recording: {collected}/{total}"
        color = (0, 255, 100)
    else:
        msg = "Press SPACE to start | Q to quit"
        color = (200, 200, 200)

    cv2.putText(
        frame,
        msg,
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )

    if state == "recording" and total > 0:
        bar_w = int((collected / total) * (w - 30))
        cv2.rectangle(frame, (15, 72), (w - 15, 82), (60, 60, 60), -1)
        cv2.rectangle(frame, (15, 72), (15 + bar_w, 82), (0, 220, 80), -1)


def main():
    os.makedirs("data", exist_ok=True)

    write_header = not os.path.exists(OUTPUT_CSV)
    csv_file = open(OUTPUT_CSV, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
        writer.writerow(header)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        csv_file.close()
        hands.close()
        raise RuntimeError(
            "Could not open webcam. Make sure camera access is enabled and no other "
            "app is using it."
        )

    for gesture in GESTURES:
        collected = 0
        state = "waiting"
        countdown_start = None

        print(f"\n-- Gesture: {gesture.upper()} --")
        print("Press SPACE when ready to record.")

        while collected < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if state == "waiting":
                draw_ui(frame, gesture, collected, SAMPLES_PER_GESTURE, "waiting")
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    state = "countdown"
                    countdown_start = time.time()
                elif key == ord("q"):
                    cap.release()
                    hands.close()
                    csv_file.close()
                    cv2.destroyAllWindows()
                    return

            elif state == "countdown":
                elapsed = time.time() - countdown_start
                remaining = max(0, COUNTDOWN_SEC - int(elapsed))
                draw_ui(
                    frame,
                    gesture,
                    collected,
                    SAMPLES_PER_GESTURE,
                    "countdown",
                    remaining,
                )
                if elapsed >= COUNTDOWN_SEC:
                    state = "recording"
                cv2.waitKey(1)

            elif state == "recording":
                draw_ui(frame, gesture, collected, SAMPLES_PER_GESTURE, "recording")
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        row = [gesture] + normalize_keypoints(hand_landmarks.landmark)
                        writer.writerow(row)
                        collected += 1
                        if collected >= SAMPLES_PER_GESTURE:
                            break
                cv2.waitKey(1)

            cv2.imshow("Data Collection", frame)

        print(f"Collected {collected} samples for '{gesture}'")

    cap.release()
    hands.close()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\nAll done! Data saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
