
import cv2
import os
import time

os.makedirs("images", exist_ok=True)

# Select camera index 1
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Camera index 1 could not be opened.")
    exit()

print("Camera index 1 opened successfully!")

# Allow camera to initialize
time.sleep(2)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Unable to receive camera frame.")
        break

    brightness = frame.mean()

    cv2.imshow("VisionPort - Camera Index 1", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.imwrite("images/camera_index1.jpg", frame)
        print("Image saved!")
        print("Brightness:", brightness)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
