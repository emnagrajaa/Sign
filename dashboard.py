import json
import os
import platform
import queue
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import joblib
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

MODEL_PATH = "models/gesture_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
REPORT_PATH = "data/report.json"

CONFIDENCE_THR = 0.55
COOLDOWN_SEC = 1.2
CAMERA_INDEX = 0
FRAME_W = 960
FRAME_H = 540

SPECIAL_GESTURES = {
    "peace": "SPACE",
    "stop": "BACKSPACE",
    "point": "CLEAR",
    "ok": "CONFIRM",
}


class SpeechEngine:
    def __init__(self, engine_type="winsapi", lang="en", rate=150):
        self.engine_type = engine_type
        self.lang = lang
        self.rate = rate
        self.last_error = ""
        self.is_speaking = False
        self._queue = queue.Queue()
        self._stop = False
        self._engine = None
        self._pygame = None
        self._is_windows = platform.system().lower() == "windows"

        if self.engine_type == "winsapi" and not self._is_windows:
            self.engine_type = "pyttsx3"

        if self.engine_type == "gtts":
            try:
                import pygame

                pygame.mixer.init()
                self._pygame = pygame
            except Exception as exc:
                self.last_error = f"gtts audio init failed: {exc}"
                self.engine_type = "winsapi" if self._is_windows else "pyttsx3"

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _setup_pyttsx3(self):
        import pyttsx3

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", self.rate)
        self._engine.setProperty("volume", 1.0)

    def speak(self, text):
        if text and text.strip():
            self._queue.put(text.strip())

    def _worker(self):
        if self.engine_type == "pyttsx3" and self._engine is None:
            try:
                self._setup_pyttsx3()
            except Exception as exc:
                self.last_error = f"pyttsx3 init failed: {exc}"

        while not self._stop:
            try:
                text = self._queue.get(timeout=0.3)
                self.is_speaking = True
                if self.engine_type == "winsapi":
                    self._speak_winsapi(text)
                elif self.engine_type == "pyttsx3":
                    self._speak_pyttsx3(text)
                else:
                    self._speak_gtts(text)
                self.is_speaking = False
            except queue.Empty:
                continue
            except Exception as exc:
                self.is_speaking = False
                self.last_error = str(exc)

    def _speak_winsapi(self, text):
        safe_text = text.replace("'", "''")
        ps_cmd = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = 0; $s.Speak('{safe_text}')"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "winsapi speech failed")

    def _speak_pyttsx3(self, text):
        if self._engine is None:
            self._setup_pyttsx3()
        self._engine.say(text)
        self._engine.runAndWait()

    def _speak_gtts(self, text):
        from gtts import gTTS

        tmp_path = None
        try:
            tts = gTTS(text=text, lang=self.lang)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            tts.save(tmp_path)
            self._pygame.mixer.music.load(tmp_path)
            self._pygame.mixer.music.play()
            while self._pygame.mixer.music.get_busy() and not self._stop:
                self._pygame.time.wait(50)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def close(self):
        self._stop = True


def normalize_keypoints(landmarks):
    kp = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    kp -= kp[0]
    scale = np.max(np.abs(kp))
    if scale > 0:
        kp /= scale
    return kp.flatten().reshape(1, -1)


class GestureDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Dashboard")
        self.root.geometry("1320x760")
        self.root.configure(bg="#1a1d24")

        self.model = joblib.load(MODEL_PATH)
        self.le = joblib.load(ENCODER_PATH)

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self.tts = SpeechEngine(engine_type="winsapi", lang="en", rate=150)

        self.cap = None
        self.running = False
        self.text_buffer = ""
        self.last_spoken = ""
        self.last_accepted = 0.0
        self.last_pred = "-"
        self.last_conf = 0.0
        self.ema_fps = 0.0

        self.mode_var = tk.StringVar(value="text")
        self.speech_style_var = tk.StringVar(value="word")

        self._build_ui()
        self.refresh_metrics()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1a1d24")
        style.configure("TLabel", background="#1a1d24", foreground="#e6e6e6")
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#ffffff")

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        controls = ttk.Frame(left)
        controls.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(controls, text="Mode:", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Radiobutton(controls, text="Text", value="text", variable=self.mode_var).pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(controls, text="Speech", value="speech", variable=self.mode_var).pack(side=tk.LEFT)

        ttk.Label(controls, text="Speech style:").pack(side=tk.LEFT, padx=(20, 4))
        ttk.Combobox(
            controls,
            textvariable=self.speech_style_var,
            values=["word", "instant"],
            width=8,
            state="readonly",
        ).pack(side=tk.LEFT)

        ttk.Button(controls, text="Start Camera", command=self.start_camera).pack(side=tk.LEFT, padx=(20, 6))
        ttk.Button(controls, text="Stop Camera", command=self.stop_camera).pack(side=tk.LEFT)

        self.video_label = tk.Label(left, bg="#111318")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        status = ttk.Frame(left)
        status.pack(fill=tk.X, pady=(8, 0))

        self.status_text = tk.StringVar(value="Ready")
        self.fps_text = tk.StringVar(value="FPS: --")
        self.conf_text = tk.StringVar(value="Confidence: --")

        ttk.Label(status, textvariable=self.status_text).pack(side=tk.LEFT, padx=4)
        ttk.Label(status, textvariable=self.fps_text).pack(side=tk.LEFT, padx=20)
        ttk.Label(status, textvariable=self.conf_text).pack(side=tk.LEFT, padx=20)

        right = ttk.Frame(main, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right.pack_propagate(False)

        ttk.Label(right, text="Metrics", style="Header.TLabel").pack(anchor="w", pady=(0, 8))

        self.metrics_text = tk.Text(
            right,
            height=20,
            wrap=tk.WORD,
            bg="#10131a",
            fg="#e6e6e6",
            insertbackground="#e6e6e6",
            relief=tk.FLAT,
            padx=10,
            pady=10,
        )
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        btns = ttk.Frame(right)
        btns.pack(fill=tk.X, pady=(8, 0))

        ttk.Button(btns, text="Refresh Metrics", command=self.refresh_metrics).pack(side=tk.LEFT)
        ttk.Button(btns, text="Speak Test", command=lambda: self.tts.speak("Speech test from dashboard")).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Save Buffer", command=self.save_buffer).pack(side=tk.LEFT)

        ttk.Label(right, text="Live Buffer", style="Header.TLabel").pack(anchor="w", pady=(14, 4))
        self.buffer_text = tk.Text(
            right,
            height=8,
            wrap=tk.WORD,
            bg="#10131a",
            fg="#ffffff",
            insertbackground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=10,
        )
        self.buffer_text.pack(fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def refresh_metrics(self):
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)

        if not os.path.exists(REPORT_PATH):
            self.metrics_text.insert(tk.END, "report.json not found.\nRun test.py first.\n")
            self.metrics_text.config(state=tk.DISABLED)
            return

        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                report = json.load(f)
        except Exception as exc:
            self.metrics_text.insert(tk.END, f"Failed to read report: {exc}\n")
            self.metrics_text.config(state=tk.DISABLED)
            return

        self.metrics_text.insert(tk.END, f"Accuracy: {report.get('accuracy', 0) * 100:.2f}%\n")
        self.metrics_text.insert(tk.END, f"Macro F1: {report.get('macro_f1', 0):.4f}\n")
        self.metrics_text.insert(tk.END, f"Weighted F1: {report.get('weighted_f1', 0):.4f}\n")
        self.metrics_text.insert(tk.END, f"Weak class threshold: {report.get('weak_class_f1_threshold', '--')}\n\n")

        weak = report.get("weak_classes", [])
        if weak:
            self.metrics_text.insert(tk.END, "Weak Classes:\n")
            for item in weak:
                self.metrics_text.insert(tk.END, f"- {item['class']}: F1={item['f1']:.4f}\n")
        else:
            self.metrics_text.insert(tk.END, "Weak Classes: none\n")

        suggestions = report.get("suggestions", [])
        if suggestions:
            self.metrics_text.insert(tk.END, "\nSuggestions:\n")
            for s in suggestions:
                self.metrics_text.insert(tk.END, f"- {s}\n")

        self.metrics_text.config(state=tk.DISABLED)

    def start_camera(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            messagebox.showerror("Camera", "Could not open webcam.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        self.running = True
        self.status_text.set("Camera running")
        self._update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status_text.set("Camera stopped")

    def _apply_gesture(self, gesture):
        action = SPECIAL_GESTURES.get(gesture)

        if action == "SPACE":
            self.text_buffer += " "
            return
        if action == "BACKSPACE":
            self.text_buffer = self.text_buffer[:-1]
            return
        if action == "CLEAR":
            self.text_buffer = ""
            return
        if action == "CONFIRM":
            if self.mode_var.get() == "speech" and self.speech_style_var.get() == "word" and self.text_buffer.strip():
                spoken = self.text_buffer.strip()
                self.tts.speak(spoken)
                self.last_spoken = spoken
            self.text_buffer = ""
            return

        self.text_buffer += gesture
        if self.mode_var.get() == "speech" and self.speech_style_var.get() == "instant":
            self.tts.speak(gesture)
            self.last_spoken = gesture

    def _draw_overlay(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 84), (20, 20, 20), -1)
        cv2.rectangle(frame, (0, h - 84), (w, h), (20, 20, 20), -1)

        cv2.putText(
            frame,
            f"Mode: {self.mode_var.get().upper()} | Speech: {self.speech_style_var.get().upper()}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            frame,
            f"Gesture: {self.last_pred.upper()}  Conf: {self.last_conf * 100:.1f}%  FPS: {self.ema_fps:4.1f}",
            (12, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 120) if self.last_conf >= CONFIDENCE_THR else (0, 140, 220),
            2,
        )

        buf_tail = self.text_buffer[-52:] if len(self.text_buffer) > 52 else self.text_buffer
        cv2.putText(
            frame,
            f"Buffer: {buf_tail}|",
            (12, h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        if self.tts.last_error:
            cv2.putText(
                frame,
                f"TTS error: {self.tts.last_error[:70]}",
                (12, h - 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (90, 150, 255),
                1,
            )

        return frame

    def _update_frame(self):
        if not self.running or self.cap is None:
            return

        start = time.perf_counter()
        ret, frame = self.cap.read()
        if not ret:
            self.status_text.set("Frame read failed")
            self.root.after(30, self._update_frame)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        now = time.time()
        cooldown_active = (now - self.last_accepted) < COOLDOWN_SEC

        self.last_pred = "-"
        self.last_conf = 0.0

        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hl, self.mp_hands.HAND_CONNECTIONS)
                kp = normalize_keypoints(hl.landmark)
                proba = self.model.predict_proba(kp)[0]
                pred_idx = int(np.argmax(proba))
                self.last_conf = float(proba[pred_idx])
                self.last_pred = self.le.inverse_transform([pred_idx])[0]

                if self.last_conf >= CONFIDENCE_THR and not cooldown_active:
                    self._apply_gesture(self.last_pred)
                    self.last_accepted = now

        elapsed = max(time.perf_counter() - start, 1e-6)
        fps = 1.0 / elapsed
        self.ema_fps = fps if self.ema_fps == 0.0 else (0.9 * self.ema_fps + 0.1 * fps)

        frame = self._draw_overlay(frame)
        self._refresh_buffer_box()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        tk_image = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=tk_image)
        self.video_label.image = tk_image

        self.fps_text.set(f"FPS: {self.ema_fps:.1f}")
        self.conf_text.set(f"Confidence: {self.last_conf * 100:.1f}%")
        self.status_text.set(
            f"Pred: {self.last_pred} | cooldown: {'on' if cooldown_active else 'off'} | speaking: {'yes' if self.tts.is_speaking else 'no'}"
        )

        self.root.after(15, self._update_frame)

    def _refresh_buffer_box(self):
        self.buffer_text.delete("1.0", tk.END)
        self.buffer_text.insert(tk.END, self.text_buffer)
        if self.last_spoken:
            self.buffer_text.insert(tk.END, f"\n\nLast spoken: {self.last_spoken}")

    def save_buffer(self):
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", f"dashboard_buffer_{int(time.time())}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.text_buffer)
        messagebox.showinfo("Saved", f"Buffer saved to {path}")

    def on_close(self):
        self.stop_camera()
        self.hands.close()
        self.tts.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureDashboard(root)
    root.mainloop()
