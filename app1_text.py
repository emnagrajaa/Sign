import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os
from datetime import datetime


MODEL_PATH = "models/gesture_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
CONFIDENCE_THR = 0.90      # minimum confidence to accept prediction
COOLDOWN_SEC = 1.5         # seconds between accepted gestures
SAVE_TRANSCRIPT = True     # save session text to a .txt file

# Map gesture labels to special actions
SPECIAL_GESTURES = {
	"peace": "SPACE",
	"stop": "BACKSPACE",
	"point": "CLEAR",
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


def normalize_keypoints(landmarks):
	kp = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
	kp -= kp[0]  # wrist as origin
	scale = np.max(np.abs(kp))
	if scale > 0:
		kp /= scale
	return kp.flatten().reshape(1, -1)  # shape (1, 63) for model input


def apply_gesture_to_buffer(gesture, buffer):
	"""
	Takes the predicted gesture and updates the text buffer.
	Returns the updated buffer string.
	"""
	action = SPECIAL_GESTURES.get(gesture)

	if action == "SPACE":
		buffer += " "
	elif action == "BACKSPACE":
		buffer = buffer[:-1]  # remove last character
	elif action == "CLEAR":
		buffer = ""
	else:
		buffer += gesture  # append gesture label as word/letter

	return buffer


def draw_hud(frame, gesture, confidence, buffer, cooldown_active, last_gesture):
	h, w = frame.shape[:2]


	cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)

	# Gesture label
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

	# Confidence percentage
	cv2.putText(
		frame,
		f"Conf: {confidence * 100:.1f}%",
		(15, 58),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.6,
		(0, 200, 255) if confidence >= CONFIDENCE_THR else (0, 80, 200),
		2,
	)

	# Confidence bar (right side of top bar)
	bar_x, bar_y, bar_w, bar_h = w - 180, 20, 160, 14
	cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
	fill = int(confidence * bar_w)
	bar_color = (0, 220, 80) if confidence >= CONFIDENCE_THR else (0, 100, 220)
	cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)

	# Cooldown indicator
	if cooldown_active:
		cv2.putText(
			frame,
			"cooldown...",
			(w - 180, 58),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(120, 120, 120),
			1,
		)

	
	cv2.rectangle(frame, (0, h - 80), (w, h), (20, 20, 20), -1)
	cv2.putText(
		frame,
		"Sentence:",
		(15, h - 55),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.55,
		(160, 160, 160),
		1,
	)

	# Show last 50 chars if sentence is long
	display_text = buffer[-50:] if len(buffer) > 50 else buffer
	cv2.putText(
		frame,
		display_text + "|",
		(15, h - 22),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.85,
		(255, 255, 255),
		2,
	)

	
	hints = "SPACE=peace  BACKSPACE=stop  CLEAR=point  Q=quit  S=save"
	cv2.putText(
		frame,
		hints,
		(15, h - 5),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.35,
		(100, 100, 100),
		1,
	)


def save_transcript(buffer):
	os.makedirs("data", exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	path = f"data/transcript_{timestamp}.txt"
	with open(path, "w", encoding="utf-8") as f:
		f.write(buffer)
	print(f"Transcript saved to {path}")
	return path


def main():
	cap = cv2.VideoCapture(0)
	text_buffer = ""
	last_accepted = 0  # timestamp of last accepted gesture
	last_gesture = ""
	gesture_label = "-"
	confidence = 0.0

	print("Running - press Q to quit, S to save transcript")

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
				# Draw landmarks
				mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

				# Predict
				kp = normalize_keypoints(hl.landmark)
				proba = model.predict_proba(kp)[0]
				pred_idx = np.argmax(proba)
				confidence = proba[pred_idx]
				gesture_label = le.inverse_transform([pred_idx])[0]

				# Accept prediction only if confident + cooldown passed
				if confidence >= CONFIDENCE_THR and not cooldown_active:
					text_buffer = apply_gesture_to_buffer(gesture_label, text_buffer)
					last_accepted = now
					last_gesture = gesture_label
					print(f"[{gesture_label}] -> '{text_buffer}'")
		else:
			# No hand detected
			gesture_label = "-"
			confidence = 0.0

		draw_hud(frame, gesture_label, confidence, text_buffer, cooldown_active, last_gesture)

		cv2.imshow("Gesture -> Text", frame)

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
		elif key == ord("s"):
			save_transcript(text_buffer)

	# Auto-save on exit if enabled
	if SAVE_TRANSCRIPT and text_buffer.strip():
		save_transcript(text_buffer)

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
