
import cv2
import time
import os

os.makedirs("images", exist_ok=True)

print("VisionPort Camera Diagnostic")
print("OpenCV version:", cv2.__version__)

# Use Apple's AVFoundation camera backend
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera successfully opened!")

# Request a standard resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

time.sleep(3)

frame = None

# Read multiple frames to allow camera initialization
for i in range(30):
    ret, current_frame = cap.read()

    if ret and current_frame is not None:
        frame = current_frame

print("Frame received:", frame is not None)

if frame is not None:
    print("Frame dimensions:", frame.shape)
    print("Minimum pixel value:", frame.min())
    print("Maximum pixel value:", frame.max())
    print("Average pixel value:", frame.mean())

    cv2.imwrite("images/camera_test.jpg", frame)
    print("Saved: images/camera_test.jpg")

    cv2.imshow("VisionPort Camera Diagnostic", frame)
    print("Press Q to close the window.")

    cv2.waitKey(0)

else:
    print("ERROR: No valid camera frames received.")

cap.release()
cv2.destroyAllWindows()
