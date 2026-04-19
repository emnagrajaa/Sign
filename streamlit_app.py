import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "gesture_model.pkl"
ENCODER_PATH = ROOT / "models" / "label_encoder.pkl"
REPORT_PATH = ROOT / "data" / "report.json"
TEXT_APP = ROOT / "app1_text.py"
SPEECH_APP = ROOT / "app2_speech.py"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Missing label encoder file: {ENCODER_PATH}")

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "MediaPipe was installed, but mp.solutions is unavailable in this runtime. "
            "Use Python 3.12 on Streamlit Cloud (add runtime.txt with python-3.12)."
        )

    if not hasattr(mp.solutions, "hands"):
        raise RuntimeError(
            "MediaPipe Hands API is unavailable in this runtime. "
            "Use Python 3.12 on Streamlit Cloud (runtime.txt: python-3.12)."
        )

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return model, label_encoder, mp_hands, mp_draw, hands


def normalize_keypoints(landmarks):
    kp = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    kp -= kp[0]
    scale = np.max(np.abs(kp))
    if scale > 0:
        kp /= scale
    return kp.flatten().reshape(1, -1)


def launch_script(script_path):
    if not script_path.exists():
        return None, f"Script not found: {script_path}"
    try:
        proc = subprocess.Popen(["python", str(script_path)], cwd=str(ROOT))
        return proc, None
    except Exception as exc:
        return None, str(exc)


def load_phase7_report():
    if not REPORT_PATH.exists():
        return None
    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def run_snapshot_inference(image_bytes):
    try:
        model, label_encoder, mp_hands, mp_draw, hands = load_artifacts()
    except Exception as exc:
        return None, None, None, str(exc)

    start = time.perf_counter()

    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return None, None, None, "Failed to decode camera image."

    frame_bgr = cv2.flip(frame_bgr, 1)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    prediction = "-"
    confidence = 0.0

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        kp = normalize_keypoints(hand_landmarks.landmark)
        proba = model.predict_proba(kp)[0]
        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])
        prediction = label_encoder.inverse_transform([pred_idx])[0]

        mp_draw.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
        )

    elapsed = max(time.perf_counter() - start, 1e-6)
    fps = 1.0 / elapsed

    overlay = frame_bgr.copy()
    h, w = overlay.shape[:2]
    cv2.rectangle(overlay, (0, 0), (w, 64), (20, 20, 20), -1)
    cv2.putText(
        overlay,
        f"Pred: {prediction.upper()}  Conf: {confidence * 100:.1f}%  FPS(est): {fps:.1f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 220, 120),
        2,
    )
    cv2.putText(
        overlay,
        "Live Snapshot Inference",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (190, 190, 190),
        1,
    )

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb, prediction, confidence, None


def render_launcher_tab():
    st.subheader("Launch Modes")
    st.write("Open existing real-time desktop apps from the browser UI.")

    if "text_proc" not in st.session_state:
        st.session_state.text_proc = None
    if "speech_proc" not in st.session_state:
        st.session_state.speech_proc = None

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("Start Text Mode"):
            proc, err = launch_script(TEXT_APP)
            if err:
                st.error(f"Text mode failed: {err}")
            else:
                st.session_state.text_proc = proc
                st.success(f"Text mode started (PID {proc.pid})")

    with c2:
        if st.button("Stop Text Mode"):
            proc = st.session_state.text_proc
            if proc and proc.poll() is None:
                proc.terminate()
                st.success("Text mode stopped")
            else:
                st.info("Text mode is not running")

    with c3:
        if st.button("Start Speech Mode"):
            proc, err = launch_script(SPEECH_APP)
            if err:
                st.error(f"Speech mode failed: {err}")
            else:
                st.session_state.speech_proc = proc
                st.success(f"Speech mode started (PID {proc.pid})")

    with c4:
        if st.button("Stop Speech Mode"):
            proc = st.session_state.speech_proc
            if proc and proc.poll() is None:
                proc.terminate()
                st.success("Speech mode stopped")
            else:
                st.info("Speech mode is not running")

    st.caption("Use the Live Snapshot tab below for browser-based confidence/FPS checks.")


def render_live_tab():
    st.subheader("Live Confidence / FPS")
    st.write("Capture a webcam snapshot in the browser and run gesture inference with confidence and estimated FPS.")

    camera_image = st.camera_input("Take a snapshot")
    if camera_image is None:
        return

    overlay_rgb, prediction, confidence, error = run_snapshot_inference(camera_image.getvalue())
    if error:
        st.error(error)
        return

    st.image(overlay_rgb, channels="RGB", caption="Inference Overlay", use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Prediction", prediction)
    with c2:
        st.metric("Confidence", f"{confidence * 100:.1f}%")


def render_metrics_tab():
    st.subheader("Weak-Class Metrics")
    report = load_phase7_report()
    if report is None:
        st.warning("report.json not found or unreadable. Run: python test.py --no-live")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Accuracy", f"{report.get('accuracy', 0.0) * 100:.2f}%")
    with c2:
        st.metric("Macro F1", f"{report.get('macro_f1', 0.0):.4f}")
    with c3:
        st.metric("Weighted F1", f"{report.get('weighted_f1', 0.0):.4f}")

    weak_classes = report.get("weak_classes", [])
    if weak_classes:
        df = pd.DataFrame(weak_classes).sort_values("f1", ascending=True)
        st.markdown("**Weak classes**")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.success("No weak classes under the configured threshold.")

    suggestions = report.get("suggestions", [])
    if suggestions:
        st.markdown("**Suggestions**")
        for item in suggestions:
            st.write(f"- {item}")


st.set_page_config(page_title="Gesture Web Dashboard", layout="wide")
st.title("Gesture Recognition Web Dashboard")
st.caption("Streamlit control center for text/speech launch, live confidence checks, and metrics.")

missing = [str(p) for p in [MODEL_PATH, ENCODER_PATH] if not p.exists()]
if missing:
    st.error("Missing required artifacts:\n" + "\n".join(missing))
    st.stop()

tab1, tab2, tab3 = st.tabs(["Launcher", "Live Snapshot", "Metrics"])
with tab1:
    render_launcher_tab()
with tab2:
    render_live_tab()
with tab3:
    render_metrics_tab()
