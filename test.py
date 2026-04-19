import argparse
import json
import os
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_CSV = "data/gestures.csv"
MODEL_PATH = "models/gesture_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
REPORT_PATH = "data/report.json"


def evaluate_model(weak_class_f1_threshold=0.95):
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Missing dataset: {DATA_CSV}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Missing label encoder: {ENCODER_PATH}")

    df = pd.read_csv(DATA_CSV)
    X = df.drop("label", axis=1).values
    y = df["label"].values

    label_encoder = joblib.load(ENCODER_PATH)
    y_encoded = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    class_names = list(label_encoder.classes_)
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    weak_classes = []
    suggestions = []
    for class_name in class_names:
        class_f1 = float(report_dict[class_name]["f1-score"])
        if class_f1 < weak_class_f1_threshold:
            weak_classes.append({"class": class_name, "f1": class_f1})
            suggestions.append(
                f"Collect 100+ new samples for '{class_name}' across varied lighting, "
                "distances, and hand orientations."
            )

    print("\n=== Offline Model Quality Check ===")
    print(f"Samples: {len(df)} | Features: {X.shape[1]} | Classes: {len(class_names)}")
    print(f"Overall accuracy: {report_dict['accuracy'] * 100:.2f}%")
    print(f"Macro F1: {report_dict['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {report_dict['weighted avg']['f1-score']:.4f}")

    print("\nPer-class F1:")
    for class_name in class_names:
        class_f1 = report_dict[class_name]["f1-score"]
        print(f"- {class_name:<12} {class_f1:.4f}")

    if weak_classes:
        print("\nWeak classes detected:")
        for item in weak_classes:
            print(f"- {item['class']}: f1={item['f1']:.4f}")
    else:
        print("\nNo weak classes below threshold.")

    os.makedirs("data", exist_ok=True)
    payload = {
        "accuracy": float(report_dict["accuracy"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "weak_class_f1_threshold": weak_class_f1_threshold,
        "weak_classes": weak_classes,
        "suggestions": suggestions,
        "confusion_matrix": cm.tolist(),
        "labels": class_names,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved report: {REPORT_PATH}")


def profile_live_fps(duration_sec=20, camera_index=0, width=960, height=540):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    model = joblib.load(MODEL_PATH)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        hands.close()
        raise RuntimeError("Could not open webcam for FPS profiling.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    start = time.perf_counter()
    ema_fps = 0.0
    frame_count = 0

    capture_ms_acc = 0.0
    mediapipe_ms_acc = 0.0
    infer_ms_acc = 0.0
    frame_ms_acc = 0.0

    print("\n=== Live FPS Profiling ===")
    print("Press Q to stop early.")

    while True:
        frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        after_capture = time.perf_counter()

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        infer_start = time.perf_counter()
        results = hands.process(rgb)
        after_mediapipe = time.perf_counter()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                kp = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                kp -= kp[0]
                scale = np.max(np.abs(kp))
                if scale > 0:
                    kp /= scale
                _ = model.predict(kp.flatten().reshape(1, -1))
        after_infer = time.perf_counter()

        frame_end = time.perf_counter()
        frame_ms = (frame_end - frame_start) * 1000.0
        fps = 1000.0 / max(frame_ms, 1e-6)
        ema_fps = fps if ema_fps == 0.0 else (0.9 * ema_fps + 0.1 * fps)

        capture_ms_acc += (after_capture - frame_start) * 1000.0
        mediapipe_ms_acc += (after_mediapipe - infer_start) * 1000.0
        infer_ms_acc += (after_infer - after_mediapipe) * 1000.0
        frame_ms_acc += frame_ms
        frame_count += 1

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 54), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"FPS: {ema_fps:5.1f}",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 120),
            2,
        )
        cv2.putText(
            frame,
            f"capture {capture_ms_acc / frame_count:4.1f}ms | mp {mediapipe_ms_acc / frame_count:4.1f}ms | infer {infer_ms_acc / frame_count:4.1f}ms",
            (12, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (185, 185, 185),
            1,
        )
        cv2.imshow("Live FPS Profiler", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if (time.perf_counter() - start) >= duration_sec:
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    if frame_count == 0:
        print("No frames captured during profiling.")
        return

    print("\nLive profiler summary:")
    print(f"- Frames captured: {frame_count}")
    print(f"- Avg total frame time: {frame_ms_acc / frame_count:.2f} ms")
    print(f"- Avg capture time: {capture_ms_acc / frame_count:.2f} ms")
    print(f"- Avg MediaPipe time: {mediapipe_ms_acc / frame_count:.2f} ms")
    print(f"- Avg model inference time: {infer_ms_acc / frame_count:.2f} ms")
    print(f"- Approx avg FPS: {1000.0 / max(frame_ms_acc / frame_count, 1e-6):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Offline Model Quality Check")
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Skip webcam FPS profiling and run only offline model checks",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=20,
        help="Live profiler duration in seconds",
    )
    parser.add_argument(
        "--weak-threshold",
        type=float,
        default=0.95,
        help="Per-class F1 threshold used to mark weak classes",
    )
    args = parser.parse_args()

    evaluate_model(weak_class_f1_threshold=args.weak_threshold)
    if not args.no_live:
        profile_live_fps(duration_sec=args.duration)
    else:
        print("Live FPS profiling skipped (--no-live).")


if __name__ == "__main__":
    main()