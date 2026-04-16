import cv2

cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ Webcam opened successfully. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to grab frame")
        break

    # Flip horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Show FPS-friendly window
    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()