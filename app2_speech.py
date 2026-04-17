import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import threading
import tempfile
import os
import queue
import subprocess
import platform


MODEL_PATH = "models/gesture_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
CONFIDENCE_THR = 0.50
COOLDOWN_SEC = 1.2

# TTS settings
TTS_ENGINE = "winsapi"    # "winsapi", "pyttsx3", or "gtts"
TTS_LANG = "en"           # "en", "fr", "ar", etc.
SPEECH_RATE = 150          # only used for pyttsx3
TEST_SPEECH_ON_START = True

# Speech mode
SPEECH_MODE = "word"

SPECIAL_GESTURES = {
	"peace": "SPACE",
	"stop": "BACKSPACE",
	"point": "CLEAR",
	"ok": "CONFIRM",      # speaks the full queued sentence
}

model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
	static_image_mode=False,
	max_num_hands=1,
	min_detection_confidence=0.7,
	min_tracking_confidence=0.7,
)


class SpeechEngine:

	def __init__(self, engine_type="gtts", lang="en", rate=150):
		self.engine_type = engine_type
		self.lang = lang
		self.rate = rate
		self.is_speaking = False
		self.last_error = ""
		self._queue = queue.Queue()
		self._stop_flag = False
		self._engine = None
		self._pygame = None
		self._can_use_windows_sapi = platform.system().lower() == "windows"

		if engine_type == "pyttsx3":
			pass
		elif engine_type == "gtts":
			try:
				import pygame

				pygame.mixer.init()
				self._pygame = pygame
			except Exception as exc:
				print(f"[TTS Warning] gTTS audio backend unavailable: {exc}")
				print("[TTS Info] Falling back to pyttsx3.")
				self.engine_type = "pyttsx3"
				self._setup_pyttsx3()
		elif engine_type == "winsapi":
			if not self._can_use_windows_sapi:
				raise ValueError("winsapi is only available on Windows")
		else:
			raise ValueError("TTS_ENGINE must be 'pyttsx3', 'gtts', or 'winsapi'")

		self._thread = threading.Thread(target=self._worker, daemon=True)
		self._thread.start()

	def speak(self, text):
		if text and text.strip():
			print(f"[TTS Queue] {text.strip()}")
			self._queue.put(text.strip())

	def _setup_pyttsx3(self):
		import pyttsx3

		self._engine = pyttsx3.init()
		self._engine.setProperty("rate", self.rate)
		self._engine.setProperty("volume", 1.0)
		print("[TTS Info] pyttsx3 initialized.")

	def _worker(self):
		if self.engine_type == "pyttsx3" and self._engine is None:
			try:
				self._setup_pyttsx3()
			except Exception as exc:
				self.last_error = f"pyttsx3 init failed: {exc}"
				print(f"[TTS Error] {self.last_error}")

		while not self._stop_flag:
			try:
				text = self._queue.get(timeout=0.5)
				self.is_speaking = True
				if self.engine_type == "pyttsx3":
					self._speak_pyttsx3(text)
				elif self.engine_type == "winsapi":
					self._speak_windows_sapi(text)
				else:
					self._speak_gtts(text)
				self.is_speaking = False
			except queue.Empty:
				continue
			except Exception as exc:
				self.is_speaking = False
				self.last_error = str(exc)
				print(f"[TTS Error] {exc}")

	def _speak_pyttsx3(self, text):
		if self._engine is None:
			self._setup_pyttsx3()
		try:
			self._engine.say(text)
			self._engine.runAndWait()
		except Exception as exc:
			if self._can_use_windows_sapi:
				print(f"[TTS Warning] pyttsx3 failed: {exc}")
				print("[TTS Info] Falling back to Windows SAPI speech.")
				self.last_error = f"pyttsx3 failed: {exc}"
				self.engine_type = "winsapi"
				self._speak_windows_sapi(text)
			else:
				raise

	def _speak_windows_sapi(self, text):
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
			raise RuntimeError(result.stderr.strip() or "Windows SAPI speech failed")

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
			while self._pygame.mixer.music.get_busy() and not self._stop_flag:
				self._pygame.time.wait(50)
		except Exception as exc:
			print(f"[TTS Warning] gTTS failed: {exc}")
			print("[TTS Info] Falling back to pyttsx3 for this and future speech.")
			self.engine_type = "winsapi" if self._can_use_windows_sapi else "pyttsx3"
			if self.engine_type == "pyttsx3" and self._engine is None:
				self._setup_pyttsx3()
			self.last_error = f"gtts failed: {exc}"
			if self.engine_type == "winsapi":
				self._speak_windows_sapi(text)
			else:
				self._speak_pyttsx3(text)
		finally:
			if tmp_path and os.path.exists(tmp_path):
				try:
					os.remove(tmp_path)
				except OSError:
					pass

	def stop(self):
		self._stop_flag = True


def audio_beep_test():
	if platform.system().lower() != "windows":
		print("[Audio Test] Beep test is only available on Windows.")
		return
	try:
		import winsound

		winsound.Beep(880, 220)
		winsound.Beep(1200, 220)
		print("[Audio Test] Beep played.")
	except Exception as exc:
		print(f"[Audio Test] Beep failed: {exc}")


def normalize_keypoints(landmarks):
	kp = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
	kp -= kp[0]
	scale = np.max(np.abs(kp))
	if scale > 0:
		kp /= scale
	return kp.flatten().reshape(1, -1)


def apply_gesture(gesture, buffer, tts, speech_mode):
	action = SPECIAL_GESTURES.get(gesture)
	spoken = None

	if action == "SPACE":
		buffer += " "
	elif action == "BACKSPACE":
		buffer = buffer[:-1]
	elif action == "CLEAR":
		buffer = ""
	elif action == "CONFIRM":
		if buffer.strip():
			tts.speak(buffer.strip())
			spoken = buffer.strip()
		buffer = ""
	else:
		buffer += gesture
		if speech_mode == "instant":
			tts.speak(gesture)
			spoken = gesture

	return buffer, spoken


def draw_hud(frame, gesture, confidence, buffer, cooldown_active, tts, last_spoken, speech_mode):
	h, w = frame.shape[:2]

	# Top bar
	cv2.rectangle(frame, (0, 0), (w, 85), (20, 20, 20), -1)

	label_color = (0, 255, 100) if not cooldown_active else (100, 100, 100)
	cv2.putText(
		frame,
		f"Gesture: {gesture.upper()}",
		(15, 30),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.8,
		label_color,
		2,
	)

	cv2.putText(
		frame,
		f"Conf: {confidence * 100:.1f}%",
		(15, 58),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.6,
		(0, 200, 255) if confidence >= CONFIDENCE_THR else (0, 80, 200),
		2,
	)

	# TTS engine + mode badge
	badge = f"{tts.engine_type.upper()} | {speech_mode.upper()} mode | lang={TTS_LANG}"
	cv2.putText(
		frame,
		badge,
		(15, 80),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.38,
		(130, 130, 130),
		1,
	)

	# Speaking indicator (top right)
	if tts.is_speaking:
		cv2.circle(frame, (w - 20, 20), 8, (0, 200, 255), -1)
		cv2.putText(
			frame,
			"speaking",
			(w - 100, 25),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(0, 200, 255),
			1,
		)

	# Confidence bar
	bar_x, bar_y = w - 180, 40
	cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 160, bar_y + 12), (60, 60, 60), -1)
	fill = int(confidence * 160)
	color = (0, 220, 80) if confidence >= CONFIDENCE_THR else (0, 100, 220)
	cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 12), color, -1)

	# Text buffer (bottom)
	cv2.rectangle(frame, (0, h - 90), (w, h), (20, 20, 20), -1)

	cv2.putText(
		frame,
		"Buffer:",
		(15, h - 68),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.5,
		(160, 160, 160),
		1,
	)
	display = buffer[-50:] if len(buffer) > 50 else buffer
	cv2.putText(
		frame,
		display + "|",
		(15, h - 42),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.82,
		(255, 255, 255),
		2,
	)

	# Last spoken
	if last_spoken:
		cv2.putText(
			frame,
			f"Last spoken: \"{last_spoken}\"",
			(15, h - 16),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(0, 200, 180),
			1,
		)

	# Last TTS error
	if tts.last_error:
		cv2.putText(
			frame,
			f"TTS error: {tts.last_error[:60]}",
			(15, h - 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.4,
			(80, 140, 255),
			1,
		)

	# Hints
	if speech_mode == "word":
		hint = "ok=SPEAK sentence  peace=SPACE  stop=BACKSPACE  point=CLEAR  T=test  M=mode  Q=quit"
	else:
		hint = "peace=SPACE  stop=BACKSPACE  point=CLEAR  T=test  M=mode  Q=quit  (instant speech)"
	cv2.putText(
		frame,
		hint,
		(15, h - 2),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.33,
		(90, 90, 90),
		1,
	)


def validate_special_gestures():
	classes = set(le.classes_)
	missing = [g for g in SPECIAL_GESTURES if g not in classes]
	if missing:
		print(f"[Warning] Special gestures not in model classes: {missing}")
		print(f"[Info] Available classes: {sorted(classes)}")


def main():
	validate_special_gestures()

	tts = SpeechEngine(
		engine_type=TTS_ENGINE,
		lang=TTS_LANG,
		rate=SPEECH_RATE,
	)
	mode = SPEECH_MODE
	print(f"TTS engine: {tts.engine_type} | lang: {TTS_LANG} | mode: {mode}")
	print("Controls: Q quit | T test speech | B beep test | M toggle instant/word mode")
	print("In WORD mode, speech happens only when you show 'ok' (CONFIRM).")

	if TEST_SPEECH_ON_START:
		tts.speak("Speech engine ready")

	cap = cv2.VideoCapture(0)
	text_buffer = ""
	last_accepted = 0
	gesture_label = "-"
	confidence = 0.0
	last_spoken = ""

	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				break

			frame = cv2.flip(frame, 1)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			results = hands.process(rgb)

			now = time.time()
			cooldown_active = (now - last_accepted) < COOLDOWN_SEC

			if results.multi_hand_landmarks:
				for hl in results.multi_hand_landmarks:
					mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

					kp = normalize_keypoints(hl.landmark)
					proba = model.predict_proba(kp)[0]
					pred_idx = np.argmax(proba)
					confidence = proba[pred_idx]
					gesture_label = le.inverse_transform([pred_idx])[0]

					if confidence >= CONFIDENCE_THR and not cooldown_active:
						text_buffer, spoken = apply_gesture(gesture_label, text_buffer, tts, mode)
						last_accepted = now
						if spoken:
							last_spoken = spoken
						print(f"[{gesture_label}] buffer='{text_buffer}'")
			else:
				gesture_label = "-"
				confidence = 0.0

			draw_hud(
				frame,
				gesture_label,
				confidence,
				text_buffer,
				cooldown_active,
				tts,
				last_spoken,
				mode,
			)

			cv2.imshow("Gesture -> Speech", frame)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
			elif key == ord("t"):
				tts.speak("Test speech")
				last_spoken = "Test speech"
			elif key == ord("m"):
				mode = "instant" if mode == "word" else "word"
				print(f"[Mode] Switched to {mode}")
				tts.speak(f"{mode} mode")
			elif key == ord("b"):
				audio_beep_test()
	finally:
		tts.stop()
		cap.release()
		cv2.destroyAllWindows()
		print("Done")


if __name__ == "__main__":
	main()
